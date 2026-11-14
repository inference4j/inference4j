# OCR document parser

`Inference4jOcrDocumentParser` is a LangChain4j `DocumentParser` that reads the text in an image with inference4j
[OCR](../../use-cases/ocr.md) (PaddleOCR PP-OCRv6), in-process. The default models are about 31 MB.

## Quick example

```java
try (var parser = new Inference4jOcrDocumentParser(OcrPipeline.builder().build())) {
    Document invoice = FileSystemDocumentLoader.loadDocument(Path.of("invoice.png"), parser);
    System.out.println(invoice.text());
}
```

The parser works with any LangChain4j document loader. To load a directory of images:

```java
PathMatcher images = FileSystems.getDefault().getPathMatcher("glob:*.{png,jpg,jpeg}");
List<Document> scans = FileSystemDocumentLoader.loadDocuments(Path.of("scans"), images, parser);
```

`loadDocuments` skips files that fail to parse, such as an image without text, and logs a warning.

## What the document contains

- **Text**: one line per detected line of text, top to bottom, then left to right.
- **Metadata**: the loader's own keys (such as `file_name`), plus:

| Key | Type | Value |
|-----|------|-------|
| `ocr_line_count` (`Inference4jOcrDocumentParser.LINE_COUNT`) | `int` | Number of text lines read |
| `ocr_mean_confidence` (`Inference4jOcrDocumentParser.MEAN_CONFIDENCE`) | `double` | Mean recognition confidence of those lines, 0 to 1 |

The confidence can be used to filter low-quality scans:

```java
Document scan = FileSystemDocumentLoader.loadDocument(path, parser);
if (scan.metadata().getDouble(Inference4jOcrDocumentParser.MEAN_CONFIDENCE) < 0.8) {
    // skip, or route to manual review
}
```

## Options

The parser has one constructor, `Inference4jOcrDocumentParser(OcrPipeline)`. Options are set on the pipeline:

| Pipeline option | Default | Description |
|-----------------|---------|-------------|
| `.minConfidence(float)` | `0` (keep every line) | Drop lines read with a lower confidence |
| `.detector(TextDetector)` | `PaddleTextDetector.small()` | Finds the lines |
| `.recognizer(TextRecognizer)` | `PaddleTextRecognizer.small()` | Reads each line |

```java
var parser = new Inference4jOcrDocumentParser(OcrPipeline.builder()
        .detector(PaddleTextDetector.tiny().build())        // ~6 MB in total, a little less accurate
        .recognizer(PaddleTextRecognizer.tiny().build())
        .minConfidence(0.5f)
        .build());
```

## Notes

- **Image formats.** Anything `javax.imageio.ImageIO` reads: PNG, JPEG, BMP and GIF with a standard JDK. Other
  input fails with `InferenceException` ("Unsupported image format").
- **Images without text.** LangChain4j documents cannot be blank, so an image without text throws
  `InferenceException` ("No text found in image").
- **Layout.** Lines are ordered by position only. Multi-column pages and tables are read row by row, and table
  and heading structure is not preserved. LangChain4j's Docling parser provides layout-aware conversion.
- **Scripts.** The default recognizer's dictionary covers Latin scripts, Chinese and Japanese. See
  [OCR](../../use-cases/ocr.md#models).
- **Lifecycle.** `close()` closes the pipeline, including its detector and recognizer.
  See [Lifecycle](index.md#lifecycle).

## Scanned PDFs

The parser reads images, not PDFs. Render each page to an image first, for example with Apache PDFBox 3, and
parse each page:

```java
List<Document> pages = new ArrayList<>();
try (PDDocument pdf = Loader.loadPDF(new File("contract.pdf"))) {
    PDFRenderer renderer = new PDFRenderer(pdf);
    for (int page = 0; page < pdf.getNumberOfPages(); page++) {
        BufferedImage image = renderer.renderImageWithDPI(page, 200);

        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(image, "png", png);
        try {
            Document document = parser.parse(new ByteArrayInputStream(png.toByteArray()));
            document.metadata().put("file_name", "contract.pdf").put("page", page + 1);
            pages.add(document);
        } catch (InferenceException e) {
            // a page without text
        }
    }
}
```

For PDFs with a text layer, use LangChain4j's `ApachePdfBoxDocumentParser`, which extracts the text directly.

## Other OCR options

| Option | Runs | Output |
|--------|-----------|---------|
| `Inference4jOcrDocumentParser` | In the JVM | Lines of plain text from an image |
| LangChain4j Docling parser | A separate `docling-serve` process | Structured documents with layout and tables, from many file formats |
| A multimodal chat model | Hosted API, or a local model server | Model-generated text; one model call per image |
