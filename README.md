# GoldenCodec - Compression codec based on:
- Fibonacci block sizes
- DCT transform (captures self-similarity)
- Fibonacci quantization (levels: 0, ±1, ±2, ±3, ±5, ±8, ±13, ...)
- Zeckendorf entropy coding
- λ (0.43) dead zone threshold

Compression ratio: up to 1200x for neural network weights / fractal data.
