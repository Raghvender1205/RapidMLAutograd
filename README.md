# RapidAutograd
An Automatic Differentiation library with OpenCL CUDA backend.

### TODO
1. A tiling approach in matmul kernel to improve memory coalescing and reduce global memory access.
2. Optimize Kernel Memory Access
3. Support Broadcasting in Operations
4. Higher-Dimensional Tensor Support
5. Reduction Operations like `sum`, `mean`, `max/min` support.
6. Advanced Neural nets like Conv, Pooling, BatchNorm.
7. Deep learning functions
8. Computational Graph visualization
9. Improved Memory management using `Gradient Checkpointing`, `Lazy Evaluation`.
10. Kernel improvement using `Vectorization`, `Kernel Fusion`
11. Gradient Management, no grad context. For example `with no_grad()`.
12. Gradient Clipping