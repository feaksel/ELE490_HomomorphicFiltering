# Handwriting OCR Readability (microsoft/trocr-large-handwritten)

| Dataset | Method | Lines | Corpus CER (%) | Corpus WER (%) | Avg preprocess (ms) | Avg OCR (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| all | Original | 38 | 12.20 | 39.44 | 23.3 | 518.2 |
| all | HF + Tone (Gray) | 38 | 12.32 | 40.49 | 660.1 | 528.1 |
| all | HSI HF + Tone | 38 | 12.45 | 41.20 | 816.9 | 520.4 |
| all | Sauvola | 38 | 12.32 | 40.14 | 102.1 | 538.8 |
| all | Zero-DCE++ | 38 | 11.35 | 38.03 | 11464.0 | 506.6 |
| all | RetinexNet | 38 | 11.04 | 36.97 | 17481.1 | 9714.7 |
| writing_jpeg | Original | 8 | 34.53 | 90.48 | 49.3 | 364.0 |
| writing_jpeg | HF + Tone (Gray) | 8 | 20.86 | 57.14 | 2132.2 | 345.8 |
| writing_jpeg | HSI HF + Tone | 8 | 24.46 | 66.67 | 2644.2 | 343.2 |
| writing_jpeg | Sauvola | 8 | 20.14 | 61.90 | 340.9 | 320.0 |
| writing_jpeg | Zero-DCE++ | 8 | 21.58 | 76.19 | 574.7 | 326.2 |
| writing_jpeg | RetinexNet | 8 | 25.18 | 76.19 | 38021.4 | 25033.2 |
| bentham | Original | 30 | 10.13 | 35.36 | 16.3 | 559.3 |
| bentham | HF + Tone (Gray) | 30 | 11.53 | 39.16 | 267.6 | 576.7 |
| bentham | HSI HF + Tone | 30 | 11.33 | 39.16 | 329.6 | 567.6 |
| bentham | Sauvola | 30 | 11.60 | 38.40 | 38.4 | 597.1 |
| bentham | Zero-DCE++ | 30 | 10.40 | 34.98 | 14367.9 | 554.7 |
| bentham | RetinexNet | 30 | 9.73 | 33.84 | 12003.6 | 5629.7 |
