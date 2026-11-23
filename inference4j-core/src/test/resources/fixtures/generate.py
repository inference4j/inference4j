# Copyright 2026 the original author or authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates the tiny ONNX models the unit tests build wrappers from.

Each fixture has the input and output names, types and ranks of the real model a wrapper loads, and a few
operators whose result a test can predict by hand. The generated ``.onnx`` files are committed, so the build
needs no Python; run this script only to change or add a fixture, then commit the regenerated files:

    python generate.py            # writes every fixture next to this script
    python generate.py out/dir    # writes them somewhere else

Requires ``onnx`` and ``numpy``. Every model is checked with ``onnx.checker.check_model(full_check=True)`` and
serialized deterministically, so regenerating unchanged fixtures gives identical bytes.

``FixtureModelsTest`` opens every fixture with the Java ONNX Runtime and checks its inputs, outputs and one
canonical result; update it with the fixtures.
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IR_VERSION = 8
OPSET = 13
PRODUCER = "inference4j-test"

FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64


def const(name, values, dtype=np.float32):
    """A Constant node producing ``values``."""
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(np.asarray(values, dtype=dtype), name + "_value"))


def model(name, nodes, inputs, outputs):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    built = helper.make_model(graph, producer_name=PRODUCER, opset_imports=[helper.make_opsetid("", OPSET)])
    built.ir_version = IR_VERSION
    onnx.checker.check_model(built, full_check=True)
    return built


def tokens(*names):
    return [helper.make_tensor_value_info(n, INT64, ["batch", "sequence"]) for n in names]


def tokens_echo(with_types):
    """logits[b, s] = ids * mask (+ 1000 * token types): each position reports the token it was given."""
    nodes = [helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"])]
    summed = "masked"
    if with_types:
        nodes += [
            const("thousand", 1000, np.int64),
            helper.make_node("Mul", ["token_type_ids", "thousand"], ["types"]),
            helper.make_node("Add", ["masked", "types"], ["summed"]),
        ]
        summed = "summed"
    nodes.append(helper.make_node("Cast", [summed], ["logits"], to=FLOAT))
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_types else [])
    return model("tokens_echo", nodes, tokens(*names),
                 [helper.make_tensor_value_info("logits", FLOAT, ["batch", "sequence"])])


def tokens_onehot(with_types):
    """last_hidden_state[b, s, :] = 10 at index ids[b, s] (0 elsewhere, all 0 when masked) (+ token type)."""
    nodes = [
        const("depth", [16], np.int64),
        const("on_off", [0, 10]),
        const("last_axis", [-1], np.int64),
        helper.make_node("OneHot", ["input_ids", "depth", "on_off"], ["onehot"], axis=-1),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], to=FLOAT),
        helper.make_node("Unsqueeze", ["mask_float", "last_axis"], ["mask_3d"]),
    ]
    if with_types:
        nodes += [
            helper.make_node("Mul", ["onehot", "mask_3d"], ["masked"]),
            helper.make_node("Cast", ["token_type_ids"], ["types_float"], to=FLOAT),
            helper.make_node("Unsqueeze", ["types_float", "last_axis"], ["types_3d"]),
            helper.make_node("Add", ["masked", "types_3d"], ["last_hidden_state"]),
        ]
    else:
        nodes.append(helper.make_node("Mul", ["onehot", "mask_3d"], ["last_hidden_state"]))
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if with_types else [])
    return model("tokens_onehot", nodes, tokens(*names),
                 [helper.make_tensor_value_info("last_hidden_state", FLOAT, ["batch", "sequence", 16])])


def tokens_passage_sum():
    """logits[b, 0] = sum of ids * mask * types: the sum of the second segment's token ids."""
    nodes = [
        helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
        helper.make_node("Mul", ["masked", "token_type_ids"], ["passage"]),
        helper.make_node("Cast", ["passage"], ["passage_float"], to=FLOAT),
        const("sequence_axis", [1], np.int64),
        helper.make_node("ReduceSum", ["passage_float", "sequence_axis"], ["logits"], keepdims=1),
    ]
    return model("tokens_passage_sum", nodes, tokens("input_ids", "attention_mask", "token_type_ids"),
                 [helper.make_tensor_value_info("logits", FLOAT, ["batch", 1])])


def depth_channel_sum():
    """predicted_depth[b, y, x] = sum over the three channels of pixel_values[b, :, y, x]."""
    nodes = [
        const("channel_axis", [1], np.int64),
        helper.make_node("ReduceSum", ["pixel_values", "channel_axis"], ["predicted_depth"], keepdims=0),
    ]
    return model("depth_channel_sum", nodes,
                 [helper.make_tensor_value_info("pixel_values", FLOAT, ["batch", 3, "height", "width"])],
                 [helper.make_tensor_value_info("predicted_depth", FLOAT, ["batch", "height", "width"])])


def sam2_encoder():
    """Three image embeddings, each the per-channel mean of pixel_values: [1, 3, 1, 1]."""
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["image_embeddings.0"], axes=[2, 3], keepdims=1),
        helper.make_node("Identity", ["image_embeddings.0"], ["image_embeddings.1"]),
        helper.make_node("Identity", ["image_embeddings.0"], ["image_embeddings.2"]),
    ]
    embedding = [1, 3, 1, 1]
    return model("sam2_encoder", nodes,
                 [helper.make_tensor_value_info("pixel_values", FLOAT, [1, 3, 1024, 1024])],
                 [helper.make_tensor_value_info("image_embeddings.%d" % i, FLOAT, embedding) for i in range(3)])


def sam2_decoder():
    """Three candidate masks scored [0.2, 0.9, 0.5] plus sum(input_boxes) / 10240.

    Every other input is summed with weight 0, so the model still consumes it and ONNX Runtime still requires
    it. The masks are 2 x 2 logits: all negative, all positive, and positive in the right column.
    """
    nodes = [const("all_axes", [0, 1, 2, 3], np.int64), const("box_axes", [0, 1, 2], np.int64)]
    terms = []
    for name, axes in [("image_embeddings.0", "all_axes"), ("image_embeddings.1", "all_axes"),
                       ("image_embeddings.2", "all_axes"), ("input_points", "all_axes")]:
        nodes.append(helper.make_node("ReduceSum", [name, axes], [name + "_sum"], keepdims=0))
        terms.append(name + "_sum")
    nodes += [
        helper.make_node("Cast", ["input_labels"], ["labels_float"], to=FLOAT),
        helper.make_node("ReduceSum", ["labels_float", "box_axes"], ["labels_sum"], keepdims=0),
        helper.make_node("ReduceSum", ["input_boxes", "box_axes"], ["boxes_sum"], keepdims=0),
    ]
    terms.append("labels_sum")
    nodes.append(helper.make_node("Sum", terms, ["unused_sum"]))
    nodes += [
        const("zero", 0),
        helper.make_node("Mul", ["unused_sum", "zero"], ["unused"]),
        const("box_scale", 1 / 10240),
        helper.make_node("Mul", ["boxes_sum", "box_scale"], ["box_term"]),
        const("base_scores", [[[0.2, 0.9, 0.5]]]),
        helper.make_node("Sum", ["base_scores", "box_term", "unused"], ["iou_scores"]),
        const("pred_masks", np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1]).reshape(1, 1, 3, 2, 2)),
    ]
    inputs = [helper.make_tensor_value_info("image_embeddings.%d" % i, FLOAT, [1, 3, 1, 1]) for i in range(3)]
    inputs += [
        helper.make_tensor_value_info("input_points", FLOAT, [1, 1, "points", 2]),
        helper.make_tensor_value_info("input_labels", INT64, [1, 1, "points"]),
        helper.make_tensor_value_info("input_boxes", FLOAT, [1, "boxes", 4]),
    ]
    return model("sam2_decoder", nodes, inputs, [
        helper.make_tensor_value_info("iou_scores", FLOAT, [1, 1, 3]),
        helper.make_tensor_value_info("pred_masks", FLOAT, [1, 1, 3, 2, 2]),
    ])


def constant_outputs(name, input_info, outputs):
    """A model that ignores its input and returns constants; the input is still declared and shape-checked."""
    nodes = [const(out, values) for out, values in outputs.items()]
    infos = [helper.make_tensor_value_info(out, FLOAT, list(np.asarray(values).shape))
             for out, values in outputs.items()]
    return model(name, nodes, [input_info], infos)


def yolov8_one_person():
    """One candidate box (cx 16, cy 16, w 20, h 20 on a 32 x 32 input), class 0 at 0.9 and class 1 at 0.1."""
    output = np.array([16, 16, 20, 20, 0.9, 0.1, 0, 0, 0]).reshape(1, 9, 1)
    return constant_outputs("yolov8_one_person",
                            helper.make_tensor_value_info("images", FLOAT, [1, 3, 32, 32]),
                            {"output0": output})


def yolo26_one_person():
    """One query: class 0 logit 3 (sigmoid 0.953), others -5; normalized box centered, a quarter wide and high."""
    return constant_outputs("yolo26_one_person",
                            helper.make_tensor_value_info("images", FLOAT, [1, 3, 32, 32]),
                            {"logits": np.array([3, -5, -5, -5, -5]).reshape(1, 1, 5),
                             "pred_boxes": np.array([0.5, 0.5, 0.25, 0.25]).reshape(1, 1, 4)})


def rtmo_people():
    """Five RTMO rows on a 32 x 32 input: dets [1, 5, 5] (x1, y1, x2, y2, score) and keypoints [1, 5, 17, 3].

    Row 0 (0.7) overlaps row 1 (0.9) and is removed by NMS; row 2 (0.8) is a second person; row 3 (0.2) falls below
    the default threshold; row 4 is the export's padding row (score 0, keypoint scores 0, junk coordinates). Keypoint
    k of rows 0 to 3 is at (3 + r + 0.5 k, 3 + k), scored 0.9 for even k and 0.1 for odd k.
    """
    dets = np.array([[2, 2, 14, 30, 0.7], [2, 2, 14, 28, 0.9], [18, 4, 30, 28, 0.8], [0, 0, 4, 4, 0.2],
                     [7, 7, 9, 9, 0]])
    keypoints = np.full((5, 17, 3), 7.0)
    for r in range(5):
        for k in range(17):
            keypoints[r, k] = [3 + r + 0.5 * k, 3 + k, 0.9 if k % 2 == 0 else 0.1] if r < 4 else [7, 7, 0]
    return constant_outputs("rtmo_people",
                            helper.make_tensor_value_info("input", FLOAT, [1, 3, 32, 32]),
                            {"dets": dets.reshape(1, 5, 5), "keypoints": keypoints.reshape(1, 5, 17, 3)})


def wav2vec2_abc():
    """CTC logits over 5 steps and a 5-token vocabulary whose argmax per step is 1, 0, 2, 0, 3."""
    logits = np.full((1, 5, 5), -10.0)
    for step, token in enumerate([1, 0, 2, 0, 3]):
        logits[0, step, token] = 10.0
    return constant_outputs("wav2vec2_abc",
                            helper.make_tensor_value_info("input_values", FLOAT, [1, "samples"]),
                            {"logits": logits})


def silero_vad():
    """output = input[:, -1] + 0.01 * state[0, 0, 0] + 0.001 * input[:, 0]; stateN = state + 1.

    The state grows by one per window, so window k scores its last sample plus 0.01 k when the state is threaded
    through; the first input sample is the context carried over from the previous window.
    """
    nodes = [
        const("last_start", [-1], np.int64),
        const("last_end", [np.iinfo(np.int64).max], np.int64),
        const("first_start", [0], np.int64),
        const("first_end", [1], np.int64),
        const("sample_axis", [1], np.int64),
        helper.make_node("Slice", ["input", "last_start", "last_end", "sample_axis"], ["last"]),
        helper.make_node("Slice", ["input", "first_start", "first_end", "sample_axis"], ["first"]),
        const("state_start", [0, 0, 0], np.int64),
        const("state_end", [1, 1, 1], np.int64),
        helper.make_node("Slice", ["state", "state_start", "state_end"], ["state_first"]),
        const("flat_shape", [1, 1], np.int64),
        helper.make_node("Reshape", ["state_first", "flat_shape"], ["state_value"]),
        const("state_weight", 0.01),
        helper.make_node("Mul", ["state_value", "state_weight"], ["state_term"]),
        const("context_weight", 0.001),
        helper.make_node("Mul", ["first", "context_weight"], ["context_term"]),
        helper.make_node("Sum", ["last", "state_term", "context_term"], ["output"]),
        helper.make_node("Cast", ["sr"], ["sr_float"], to=FLOAT),
        const("zero", 0),
        helper.make_node("Mul", ["sr_float", "zero"], ["sr_term"]),
        const("one", 1),
        helper.make_node("Sum", ["state", "one", "sr_term"], ["stateN"]),
    ]
    return model("silero_vad", nodes, [
        helper.make_tensor_value_info("input", FLOAT, [1, "samples"]),
        helper.make_tensor_value_info("state", FLOAT, [2, 1, 128]),
        helper.make_tensor_value_info("sr", INT64, []),
    ], [
        helper.make_tensor_value_info("output", FLOAT, [1, 1]),
        helper.make_tensor_value_info("stateN", FLOAT, [2, 1, 128]),
    ])


def clip_vision():
    """image_embeds[b, c] = mean of pixel_values[b, c, :, :]."""
    nodes = [helper.make_node("ReduceMean", ["pixel_values"], ["image_embeds"], axes=[2, 3], keepdims=0)]
    return model("clip_vision", nodes,
                 [helper.make_tensor_value_info("pixel_values", FLOAT, ["batch", 3, "height", "width"])],
                 [helper.make_tensor_value_info("image_embeds", FLOAT, ["batch", 3])])


def clip_text():
    """text_embeds[b, k] = how many unmasked tokens have id k, for k in 0..2; other ids count nowhere."""
    nodes = [
        const("depth", [3], np.int64),
        const("on_off", [0, 1]),
        const("last_axis", [-1], np.int64),
        const("sequence_axis", [1], np.int64),
        helper.make_node("OneHot", ["input_ids", "depth", "on_off"], ["onehot"], axis=-1),
        helper.make_node("Cast", ["attention_mask"], ["mask_float"], to=FLOAT),
        helper.make_node("Unsqueeze", ["mask_float", "last_axis"], ["mask_3d"]),
        helper.make_node("Mul", ["onehot", "mask_3d"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "sequence_axis"], ["text_embeds"], keepdims=0),
    ]
    return model("clip_text", nodes, tokens("input_ids", "attention_mask"),
                 [helper.make_tensor_value_info("text_embeds", FLOAT, ["batch", 3])])


def classifier_nhwc():
    """Returns its 1 x 32 x 32 x 3 (channels last) input unchanged, so each input value is a class score."""
    nodes = [helper.make_node("Identity", ["input"], ["scores"])]
    return model("classifier_nhwc", nodes,
                 [helper.make_tensor_value_info("input", FLOAT, [1, 32, 32, 3])],
                 [helper.make_tensor_value_info("scores", FLOAT, [1, 32, 32, 3])])


FIXTURES = {
    "tokens-echo": lambda: tokens_echo(False),
    "tokens-echo-types": lambda: tokens_echo(True),
    "tokens-onehot": lambda: tokens_onehot(False),
    "tokens-onehot-types": lambda: tokens_onehot(True),
    "tokens-passage-sum": tokens_passage_sum,
    "depth-channel-sum": depth_channel_sum,
    "sam2-encoder": sam2_encoder,
    "sam2-decoder": sam2_decoder,
    "yolov8-one-person": yolov8_one_person,
    "yolo26-one-person": yolo26_one_person,
    "rtmo-people": rtmo_people,
    "wav2vec2-abc": wav2vec2_abc,
    "silero-vad": silero_vad,
    "clip-vision": clip_vision,
    "clip-text": clip_text,
    "classifier-nhwc": classifier_nhwc,
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out, exist_ok=True)
    for name, build in FIXTURES.items():
        path = os.path.join(out, name + ".onnx")
        with open(path, "wb") as f:
            f.write(build().SerializeToString(deterministic=True))
        print("%-22s %5d bytes" % (name, os.path.getsize(path)))


if __name__ == "__main__":
    main()
