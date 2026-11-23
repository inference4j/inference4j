# Pose Detection

Find the people in an image and the position of their body joints (also called pose estimation or keypoint
detection). Each person comes back as a `Pose`: the person's box and score, plus 17 keypoints (nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles) in the image's pixel coordinates.

Typical uses: counting exercise repetitions and checking form from joint angles, posture and ergonomics scoring,
sports and dance analytics, and storing skeletons instead of images where privacy matters.

## Quick example

```java
try (PoseDetector detector = RtmoPoseDetector.builder().build()) {      // RTMO-s, COCO only
    List<Pose> poses = detector.detect(Path.of("gym.jpg"));
    Keypoint wrist = poses.get(0).keypoint(CocoJoint.LEFT_WRIST);
}
```

## Choose a model

| Preset | Model | Weights license | Training data | COCO keypoint AP (authors) | ONNX size | CPU latency* |
|--------|-------|-----------------|---------------|----------------------------|-----------|--------------|
| `RtmoPoseDetector.s()` (default) | RTMO-s | **Apache-2.0** | COCO | 67.7 | 40 MB | 43 ms |
| `RtmoPoseDetector.m()` | RTMO-m | **Apache-2.0** | COCO | 70.9 | 89 MB | 91 ms |
| `RtmoPoseDetector.l()` | RTMO-l | **Apache-2.0** | COCO | 72.4 | 176 MB | 155 ms |

\*Mean per image over seven COCO photos (640 × 640 input) on a laptop CPU with four ONNX Runtime threads, measured
with inference4j. AP figures come from the [RTMO authors](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo)
(COCO val2017); inference4j has not re-measured them.

- **Start with RTMO-s**: the smallest and fastest.
- **RTMO-m or RTMO-l** when accuracy matters more than speed, for example small or distant people.

RTMO is a one-stage model: one run finds every person and their joints, so the cost does not grow with the number of
people in the image.

## Full example

The angle at a joint is the building block of repetition counters and posture checks:

```java
import io.github.inference4j.vision.CocoJoint;
import io.github.inference4j.vision.ImageAnnotator;
import io.github.inference4j.vision.Pose;
import io.github.inference4j.vision.PoseDetector;
import io.github.inference4j.vision.RtmoPoseDetector;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

public class PoseDetection {
    public static void main(String[] args) throws Exception {
        BufferedImage image = ImageIO.read(Path.of("squat.jpg").toFile());

        try (PoseDetector detector = RtmoPoseDetector.builder().build()) {
            List<Pose> poses = detector.detect(image);

            for (Pose pose : poses) {
                float knee = pose.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE);
                System.out.printf("person %.0f%%: left knee at %.0f degrees%n",
                    pose.person().confidence() * 100, knee);
            }

            ImageAnnotator.save(ImageAnnotator.annotatePoses(image, poses), Path.of("squat-pose.png"));
        }
    }
}
```

## Processing

The models are the MMPose RTMO checkpoints exported to ONNX with MMDeploy, which puts box and keypoint decoding and
non-maximum suppression inside the model. The image is scaled to fit the 640 × 640 input with OpenCV's bilinear
resize, placed at the top left and padded with gray, and passed as BGR with values 0–255, as for YOLOX. On seven COCO
photos, inference4j returns the same people as the exported model run with OpenCV pre-processing in Python: person
scores within 1e-4, boxes within 0.4 pixels, confident keypoints within 1.4 pixels.

## Drawing skeletons

`ImageAnnotator.annotatePoses` draws each pose on a copy of the image: a line for each bone of the COCO skeleton and a
dot for each joint. Joints scoring below 0.3, and the bones that touch them, are not drawn.

```java
BufferedImage annotated = ImageAnnotator.annotatePoses(image, poses);

// Custom bones, minimum joint confidence, color and line width
BufferedImage custom = ImageAnnotator.annotatePoses(image, poses, CocoJoint.edges(), 0.5f, Color.ORANGE, 3f);
```

Person boxes are not drawn; add them with `ImageAnnotator.annotate(image, boxes)` using
`poses.stream().map(p -> p.person().box()).toList()`.

## Licenses and training data

The wrapper code and the RTMO code and weights are Apache-2.0. For pose models, the training data matters as well:
many published pose checkpoints are trained on a mix of datasets, some of which come with research-only or
non-commercial terms.

The presets use the RTMO checkpoints trained on **COCO only** (the backbone starts from YOLOX weights, also trained on
COCO). They are not the RTMO "body7" checkpoints, which add six more datasets (among them AI Challenger, CrowdPose,
Halpe and MPII) and score higher.

COCO images come from Flickr under their owners' Creative Commons licenses, and the COCO annotations are
CC BY 4.0. This is the same bar as the Apache-2.0 object detectors.

This page is not legal advice; check the licenses for your use.

## Bring your own weights

Any RTMO export with the same inputs and outputs loads through `modelId` and `modelSource`: for example a checkpoint
fine-tuned on your own data, or one trained on datasets you have a commercial license for.

```java
// A local directory with model.onnx
try (PoseDetector detector = RtmoPoseDetector.builder()
        .modelId("rtmo-s-custom")
        .modelSource(new LocalModelSource(Path.of("/models")))      // loads /models/rtmo-s-custom/model.onnx
        .build()) {
    detector.detect(Path.of("image.jpg"));
}

// Your own HuggingFace repository with model.onnx
try (PoseDetector detector = RtmoPoseDetector.builder().modelId("my-org/rtmo-m-finetuned").build()) {
    detector.detect(Path.of("image.jpg"));
}
```

The model must take a BGR image of shape `[1, 3, size, size]` with raw 0–255 values and return `dets` `[1, N, 5]`
and `keypoints` `[1, N, 17, 3]`, as the MMPose/MMDeploy export does. A model with a fixed input size uses that size;
otherwise set it with `.inputSize(int)`.

## Builder options

| Method | Type | Description |
|--------|------|-------------|
| `.modelId(String)` | `String` | HuggingFace model ID; defaults to `inference4j/rtmo-s-coco` |
| `.modelSource(ModelSource)` | `ModelSource` | Model resolution strategy; defaults to `HuggingFaceModelSource` |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | ONNX Runtime session config |
| `.inputSize(int)` | `int` | Side of the square model input. A model with a fixed input size uses it, and a different value fails at `build()`; otherwise defaults to `640` |
| `.confidenceThreshold(float)` | `float` | Minimum person score; defaults to `0.5`. Keypoint scores never drop a person |
| `.iouThreshold(float)` | `float` | IoU above which a less confident person is removed; defaults to `0.65`, which the model already applies, so only a lower value changes the result |
| `.maxPoses(int)` | `int` | Most poses returned per image, the most confident first; defaults to `50`, the most the exported models return |

The person threshold can also be set per call: `detector.detect(image, 0.3f)`.

The exported models run non-maximum suppression themselves and return at most 50 people with a score above 0.1, so
thresholds below 0.1 have no effect.

## Result type

`Pose` is a record with:

| Member | Type | Description |
|--------|------|-------------|
| `person()` | `Detection` | The person's box and score, labeled `person` |
| `keypoints()` | `List<Keypoint>` | Every joint of the model's format, in its order, whatever its score |
| `keypoint(Joint)` | `Keypoint` | One joint, for example `pose.keypoint(CocoJoint.LEFT_KNEE)` |
| `angle(Joint a, Joint b, Joint c)` | `float` | Angle at `b` between the lines to `a` and to `c`, in degrees (0 to 180) |

`Keypoint` is a record with `joint()`, `x()`, `y()` and `confidence()`, and `point()` for a `Point`.

Coordinates are in the original image's pixels. A joint the model cannot see is still returned, with a low
confidence and a guessed position that may lie outside the image, so check `confidence()` before using a joint (0.3
to 0.5 is a common cut-off).

`CocoJoint` is an enum of the 17 COCO joints, in the COCO order (`NOSE` = 0 to `RIGHT_ANKLE` = 16). "Left" and
"right" are the person's own sides, not the viewer's. `CocoJoint.edges()` lists the 19 bones of the COCO skeleton.
`CocoJoint` implements the `Joint` interface, so models with other keypoint formats can add their own enum later.

## Swap contract

Every `PoseDetector` follows the same contract:

- one `Pose` per person, sorted by descending person confidence;
- the person box is in pixel coordinates of the original image, inside the image, labeled `person`;
- keypoints are in pixel coordinates of the original image and may fall outside it;
- every pose has every joint of the model's format, in its order; only the person confidence filters;
- `Path` and `BufferedImage` inputs give the same poses for the same image.

## Spring Boot

The starter registers a `PoseDetector` bean when `inference4j.vision.pose-detector.enabled=true`; `model-id`
and `confidence-threshold` are optional. See the [Spring Boot guide](../guides/spring-boot.md#pose-detection).

## Tips

- **Type variables as `PoseDetector`** to keep the model swappable.
- **Check keypoint confidence** before computing angles: a hidden knee gives a meaningless angle.
- **Video**: decode frames yourself and pass each as a `BufferedImage`. The detector is thread-safe and can be shared,
  but it does not track people from one frame to the next.
- Pose detection is not a safety or medical device; fall detection and similar uses need more than single-frame
  skeletons.
