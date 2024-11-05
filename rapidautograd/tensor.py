import numpy as np
import pyopencl as cl
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ["PYOPENCL_CTX"] = "0"


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
        return f'Tensor(data={self.data}, grad={self.grad})'

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

# Compile kernels
try:
    add_program = cl.Program(context, add_kernel_code).build(options='-w')
    multiply_program = cl.Program(context, multiply_kernel_code).build(options='-w')
except cl.ProgramBuildFailure as e:
    print("Build failed:", e)
except Exception as e:
    print("An error occurred:", e)


####################
### Ops
###################
def add(tensor_a, tensor_b):
    assert tensor_a.data.shape == tensor_b.data.shape, "Shapes must match"
    num_elements = tensor_a.data.size // 4
    result = Tensor(np.zeros_like(tensor_a.data), requires_grad=tensor_a.requires_grad or tensor_b.requires_grad)

    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tensor_a.data.view(np.float32))
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tensor_b.data.view(np.float32))
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, tensor_a.data.nbytes)

    add_program.add_kernel(queue, (num_elements,), None, a_buf, b_buf, c_buf, np.int32(num_elements))
    cl.enqueue_copy(queue, result.data.view(np.float32), c_buf).wait()

    if result.requires_grad:
        def grad_fn(grad):
            if tensor_a.requires_grad:
                tensor_a.backward(grad)
            if tensor_b.requires_grad:
                tensor_b.backward(grad)
        result.set_grad_fn(grad_fn)

    return result

def multiply(tensor_a, tensor_b):
    assert tensor_a.data.shape == tensor_b.data.shape, "Shapes must match"
    num_elements = tensor_a.data.size // 4
    result = Tensor(np.zeros_like(tensor_a.data), requires_grad=tensor_a.requires_grad or tensor_b.requires_grad)

    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tensor_a.data.view(np.float32))
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tensor_b.data.view(np.float32))
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, tensor_a.data.nbytes)

    multiply_program.multiply_kernel(queue, (num_elements,), None, a_buf, b_buf, c_buf, np.int32(num_elements))
    cl.enqueue_copy(queue, result.data.view(np.float32), c_buf).wait()

    if result.requires_grad:
        def grad_fn(grad):
            if tensor_a.requires_grad:
                tensor_a.backward(grad * tensor_b.data)
            if tensor_b.requires_grad:
                tensor_b.backward(grad * tensor_a.data)
        result.set_grad_fn(grad_fn)

    return result