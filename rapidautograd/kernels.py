import os
import pyopencl as cl

os.environ["PYOPENCL_CTX"] = '0'


add_kernel_code = """
__kernel void add_kernel(__global const float4 *a, __global const float4 *b, __global float4 *c, int num_elements) {
    int idx = get_global_id(0);
    if (idx < num_elements) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

multiply_kernel_code = """
__kernel void multiply_kernel(__global const float4 *a, __global const float4 *b, __global float4 *c, int num_elements) {
    int idx = get_global_id(0);
    if (idx < num_elements) {
        c[idx] = a[idx] * b[idx];
    }
}
"""

subtract_kernel_code = """
__kernel void subtract_kernel(__global const float4 *a, __global const float4 *b, __global float4 *c, int num_elements) {
    int idx = get_global_id(0);
    if (idx < num_elements) {
        c[idx] = a[idx] - b[idx];
    }
}
"""

matmul_kernel_code = """
__kernel void matmul_kernel(__global const float *A, __global const float *B, __global float *C, const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""

# Initialize OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Compile kernels
try:
    add_program = cl.Program(context, add_kernel_code).build(options='-w')
    multiply_program = cl.Program(context, multiply_kernel_code).build(options='-w')
    subtract_program = cl.Program(context, subtract_kernel_code).build(options='-w')
    matmul_program = cl.Program(context, matmul_kernel_code).build(options='-w')
except cl.Error as e:
    print("Build failed:", e)
except Exception as e:
    print("An error occurred:", e)
