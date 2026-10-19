# Image Segmentation

Get the exact outline of an object, pixel by pixel. You point at the object with a click or a box, and SAM 2.1 (Segment Anything Model) returns its mask. It works on any kind of object, because it was never trained on a fixed list of classes.

Typical uses:

- **Background removal and cut-outs:** product photos, profile pictures, stickers.
- **Precise masks from a detector:** an [object detector](object-detection.md) gives rectangles, and SAM turns each one into an exact outline. Use this to blur only the person, or to measure an object's area.
- **Annotation tools:** a user clicks an object to label it, then clicks again to fix the mask.

## Quick example

```java
try (var sam = Sam2Segmenter.builder().build()) {
    SegmentationMask mask = sam.segment(Path.of("cat.jpg"), Prompt.point(160, 160));
}
```

## Encode once, prompt many times

SAM runs in two steps. The **encoder** reads the image, which is the slow step. The **decoder** turns a prompt into a mask, which is fast. Encode each image once and reuse the result for every prompt:

```java
try (var sam = Sam2Segmenter.builder().build()) {
    BufferedImage image = ImageIO.read(new File("street.jpg"));
    ImageFeatures features = sam.encode(image);      // slow: once per image

    SegmentationMask car = sam.segment(features, Prompt.point(420, 310));      // fast
    SegmentationMask person = sam.segment(features, Prompt.point(120, 260));  // fast
}
```

On a laptop CPU, encoding takes well under a second and each prompt takes tens of milliseconds, so a UI can respond to every click.

## Detector + SAM: a mask for every object

A detector finds objects but only gives rectangles. Pass each rectangle to SAM as a box prompt to get each object's outline:

```java
import io.github.inference4j.vision.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;

public class CutOutPeople {
    public static void main(String[] args) throws Exception {
        BufferedImage image = ImageIO.read(new File("street.jpg"));

        try (var detector = RtDetrDetector.builder().build();
             var sam = Sam2Segmenter.builder().build()) {

            ImageFeatures features = sam.encode(image);
            int i = 0;
            for (Detection d : detector.detect(image)) {
                if (!d.label().equals("person")) {
                    continue;
                }
                SegmentationMask mask = sam.segment(features, Prompt.box(d.box()));
                ImageAnnotator.save(mask.applyTo(image), Path.of("person-" + i++ + ".png"));
            }
        }
    }
}
```

## Prompts

All coordinates are pixels in the original image. The wrapper scales them to the model's input size for you.

```java
Prompt.point(320, 240)                                // one click on the object
Prompt.point(320, 240).withPoint(360, 260)            // more clicks on the same object
Prompt.point(320, 240).withBackground(300, 400)       // exclude a region the mask took by mistake
Prompt.box(50, 40, 410, 380)                          // a box around the object
Prompt.box(detection.box())                           // a detector's box
Prompt.box(detection.box()).withPoint(220, 200)       // a box plus a hint inside it
```

A background click is how users correct a mask: if a click on a person also selects the bike, add a background click on the bike.

## Several candidate masks

A single click is ambiguous: a click on a shirt could mean the shirt or the whole person. SAM proposes three masks at different granularities and scores each one. `segment` returns the best; `segmentCandidates` returns all three, best first, so a UI can let the user pick:

```java
List<SegmentationMask> options = sam.segmentCandidates(features, Prompt.point(320, 240));
```

## Using the mask

```java
mask.applyTo(image)                  // cut-out: transparent outside the mask (save as PNG)
mask.overlay(image, Color.MAGENTA)   // highlight the object in place
mask.toImage()                       // black and white mask image
mask.contains(x, y)                  // is this pixel part of the object?
mask.area()                          // number of pixels in the mask
mask.boundingBox()                   // tightest box around the mask, if not empty
mask.score()                         // the model's predicted IoU for this mask, 0 to 1
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/sam2.1-hiera-tiny` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config, applied to both the encoder and the decoder |

## Result type

`SegmentationMask` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `mask()` | `boolean[][]` | Per-pixel membership as `[height][width]` |
| `width()` | `int` | Mask width, always matching the input image |
| `height()` | `int` | Mask height, always matching the input image |
| `score()` | `float` | Predicted IoU of this mask with the true object |

Plus `contains(x, y)`, `area()`, `isEmpty()`, `boundingBox()`, `toImage()`, `applyTo(image)` and `overlay(image, color)`.

## How SAM 2.1 works

The encoder is a Hiera vision transformer. It stretches the image to 1024×1024 (aspect ratio not preserved, as SAM 2 was trained), normalizes it with ImageNet mean and standard deviation, and produces feature maps at three resolutions. Those maps are what `ImageFeatures` holds, about 17 MB per image.

The decoder embeds the prompt, attends to the feature maps, and predicts three 256×256 mask logit maps with a predicted IoU each. The wrapper upsamples each map to 1024×1024 and then to the original image size, and marks pixels with a positive logit as part of the object.

## Available models

| Model | Parameters | Size | Notes |
|-------|-----------|------|-------|
| `inference4j/sam2.1-hiera-tiny` | ~39M | ~155 MB | Default. Apache-2.0 (Meta) |

## Hardware acceleration

```java
try (var sam = Sam2Segmenter.builder()
        .sessionOptions(opts -> opts.addCoreML())
        .build()) {
    sam.segment(Path.of("cat.jpg"), Prompt.point(160, 160));
}
```

## Tips

- Keep `ImageFeatures` only while you are prompting that image. Each one holds about 17 MB.
- Use `ImageFeatures` only with the segmenter that created it.
- Save cut-outs as PNG. JPEG has no alpha channel, so the transparent background turns black.
- Tracking an object through a video needs SAM 2's memory modules, which this export does not include. See [#137](https://github.com/inference4j/inference4j/issues/137).
