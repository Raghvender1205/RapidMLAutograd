import numpy as np
import pyopencl as cl
import os
import warnings

from rapidautograd.memorypool import MemoryPool

warnings.filterwarnings("ignore")

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ["PYOPENCL_CTX"] = "0"

memory_pool = MemoryPool()

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        if self.data.size % 4 != 0:
            padding = 4 - self.data.size % 4
            self.data = np.pad(self.data, (0, padding), 'constant', constant_values=(0,)) 
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn

    def backward(self, grad=None):
        if self.requires_grad:
            if grad is None and self.grad is None:
                self.grad = np.ones_like(self.data)
            elif grad is not None:
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

            if self._grad_fn is not None:
                self._grad_fn(self.grad)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f'RapidTensor(data={self.data}, grad={self.grad})'

##################### 
### Kernels
#####################
# Initialize OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

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


# Compile kernels
try:
    add_program = cl.Program(context, add_kernel_code).build(options='-w')
    multiply_program = cl.Program(context, multiply_kernel_code).build(options='-w')
    subtract_program = cl.Program(context, subtract_kernel_code).build(options='-w')
    matmul_program = cl.Program(context, matmul_kernel_code).build(options='-w')
except cl.ProgramBuildFailure as e:
    print("Build failed:", e)
except Exception as e:
    print("An error occurred:", e)


####################
### Ops
###################

# Operations using memory pooling
def operate_with_pooling(tensor_a, tensor_b, operation_kernel, program):
    num_elements = tensor_a.data.size // 4
    size = tensor_a.data.nbytes
    a_buf = memory_pool.allocate(context, size)
    b_buf = memory_pool.allocate(context, size)
    c_buf = memory_pool.allocate(context, size)

    # Fill buffers
    cl.enqueue_copy(queue, a_buf, tensor_a.data.view(np.float32))
    cl.enqueue_copy(queue, b_buf, tensor_b.data.view(np.float32))
    operation_kernel(queue, (num_elements,), None, a_buf, b_buf, c_buf, np.int32(num_elements))
    result = Tensor(np.zeros_like(tensor_a.data), requires_grad=tensor_a.requires_grad or tensor_b.requires_grad)

    cl.enqueue_copy(queue, result.data.view(np.float32), c_buf).wait()
    
    memory_pool.free(a_buf)
    memory_pool.free(b_buf)
    memory_pool.free(c_buf)

    return result

# Operations
def add(tensor_a, tensor_b):
    assert tensor_a.data.shape == tensor_b.data.shape, "Shapes must match"
    return operate_with_pooling(tensor_a, tensor_b, add_program.add_kernel, tensor_a.data.size // 4)

def multiply(tensor_a, tensor_b):
    assert tensor_a.data.shape == tensor_b.data.shape, "Shapes must match"
    return operate_with_pooling(tensor_a, tensor_b, multiply_program.multiply_kernel, tensor_a.data.size // 4)

def subtract(tensor_a, tensor_b):
    assert tensor_a.data.shape == tensor_b.data.shape, "Shapes must match"
    return operate_with_pooling(tensor_a, tensor_b, subtract_program.subtract_kernel, tensor_a.data.size // 4)

def matmul(tensor_a, tensor_b):
    # Assume both tensors are square matrices
    N = int(np.sqrt(tensor_a.data.size))
    assert tensor_a.data.size == tensor_b.data.size, "Tensors must be square matrices of the same size"
    assert N * N == tensor_a.data.size, "Tensor size must be a perfect square"

    # Allocate result tensor and buffers
    result = Tensor(np.zeros_like(tensor_a.data), requires_grad=tensor_a.requires_grad or tensor_b.requires_grad)
    a_buf = memory_pool.allocate(context, tensor_a.data.nbytes)
    b_buf = memory_pool.allocate(context, tensor_b.data.nbytes)
    c_buf = memory_pool.allocate(context, result.data.nbytes)

    # Copy data to buffers
    cl.enqueue_copy(queue, a_buf, tensor_a.data)
    cl.enqueue_copy(queue, b_buf, tensor_b.data)

    # Execute kernel with 2D global size
    global_size = (N, N)
    matmul_program.matmul_kernel(queue, global_size, None, a_buf, b_buf, c_buf, np.int32(N))

    # Copy result back to host
    cl.enqueue_copy(queue, result.data, c_buf).wait()

    # Free buffers
    memory_pool.free(a_buf)
    memory_pool.free(b_buf)
    memory_pool.free(c_buf)

    return result
