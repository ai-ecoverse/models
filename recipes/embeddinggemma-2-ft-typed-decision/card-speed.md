| Runtime | Variant | Three-question request |
|---|---|---|
| onnxruntime-node 1.30.0, CPU (M4 Max) | `fp32` / `q8` | 123 / 166 ms |
| onnxruntime-node 1.30.0, WebGPU (M4 Max) | `fp32` / `q8` | 67 / 58 ms |
| onnxruntime-web 1.30.0, WebGPU on Dawn (`webgpu` 0.6.2, Node 26, M4 Max) | `fp32` / `q8` | 69 / 58 ms |
| onnxruntime-web 1.30.0, WASM (Node 26, M4 Max) | `fp32` / `q8` | 709 / 872 ms |
| onnxruntime 1.31.0 Python, CPU (M4 Max) | `fp32` / `q8` | 111 / 164 ms |
| Kev-0.8B through ortllama, host CPU / host WebGPU (M4 Max, ortllama's README) | `q8f32` | 70 / 75 ms |
