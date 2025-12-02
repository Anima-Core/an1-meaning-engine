# Example Results - AN1 Meaning Engine Frozen Sender

Results from running the experiment on an H100 GPU:

## Summary Table

| Metric | Value |
|--------|-------|
| Teacher accuracy | 88.04% |
| AN1 accuracy | 70.93% |
| Teacher latency per example (ms) | 0.0696 |
| AN1 latency per example (ms) | 0.0019 |
| Speedup (x) | 36.63 |
| FLOP reduction (x) | 1370.6 |

## Speedup by Batch Size

| Batch Size | Teacher (ms) | AN1 (ms) | Speedup (x) |
|------------|--------------|----------|-------------|
| 64 | 0.0721 | 0.0021 | 34.33 |
| 256 | 0.0696 | 0.0019 | 36.63 |

## Notes

- **Teacher accuracy**: Measured using full ResNet18 on CIFAR-10 images
- **AN1 accuracy**: Measured using only AN1 head on precomputed 64-dim headers (no teacher call)
- **Teacher latency**: Measured using full ResNet18 on images with batch_size=256
- **AN1 latency**: Measured using only AN1 head on precomputed headers with batch_size=256
- **Header extraction**: Headers are precomputed from ResNet18 layer1 output (after conv1, bn1, relu, maxpool, layer1)
- **AN1 Head**: 64 → 1024 → 512 → 128 → 10 MLP
- Both latency measurements use the same warmup (10 batches) and timed batches (50 batches)
