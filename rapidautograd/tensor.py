import numpy as np
import pyopencl as cl
from rapidautograd.memorypool import MemoryPool
from rapidautograd.kernels import (
    add_program,
    subtract_program,
    matmul_program,
    multiply_program,
    context,
    queue,
)

memory_pool = MemoryPool()


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        if self.data.size % 4 != 0:
            padding = 4 - self.data.size % 4
            self.data = np.pad(
                self.data, (0, padding), "constant", constant_values=(0,)
            )
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.creator:
            self.creator.backward(grad)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def add(self, other):
        return add(self, other)

    def multiply(self, other):
        return multiply(self, other)

    def subtract(self, other):
        return subtract(self, other)

    def matmul(self, other):
        return matmul(self, other)

    def transpose(self):
        # This implementation assumes a 2D tensor for simplicity.
        if self.data.ndim != 2:
            raise ValueError("transpose currently supports 2D matrices only.")
        transposed_data = np.transpose(self.data)
        return Tensor(transposed_data, requires_grad=self.requires_grad)


###############
# TensorOps
##############
# Define generic tensor_operation function
def tensor_operation(
    tensor_a: Tensor, tensor_b: Tensor, operation_kernel, is_matmul=False
):
    if is_matmul:
        # Ensure both tensors are 2D and properly aligned
        if tensor_a.data.ndim != 2 or tensor_b.data.ndim != 2:
            raise ValueError(
                "Matmul operation requires both tensors to be 2D matrices."
            )
        if tensor_a.data.shape[1] != tensor_b.data.shape[0]:
            raise ValueError(
                "Matrix multiplication requires shape alignment: a's columns must match b's rows."
            )
        N = tensor_a.data.shape[0]  # Number of rows in tensor_a
        M = tensor_b.data.shape[1]  # Number of columns in tensor_b
        # Assuming square matrices for simplicity
        if (
            tensor_a.data.shape[0] != tensor_a.data.shape[1]
            or tensor_b.data.shape[0] != tensor_b.data.shape[1]
        ):
            raise ValueError("Matmul currently only supports square matrices.")
    else:
        # For element-wise operations, ensure tensors are flattened
        if tensor_a.data.ndim > 1 or tensor_b.data.ndim > 1:
            raise ValueError(
                "Element-wise operations currently only support 1D tensors."
            )
        N = tensor_a.data.size // 4  # Assuming float4

    result_data = np.empty_like(tensor_a.data)
    a_buf = memory_pool.allocate(context, tensor_a.data.nbytes)
    b_buf = memory_pool.allocate(context, tensor_b.data.nbytes)
    c_buf = memory_pool.allocate(context, result_data.nbytes)

    cl.enqueue_copy(queue, a_buf, tensor_a.data).wait()
    cl.enqueue_copy(queue, b_buf, tensor_b.data).wait()

    if is_matmul:
        global_size = (N, N)
        operation_kernel(queue, global_size, None, a_buf, b_buf, c_buf, np.int32(N))
    else:
        operation_kernel(queue, (N,), None, a_buf, b_buf, c_buf, np.int32(N))

    cl.enqueue_copy(queue, result_data, c_buf).wait()
    memory_pool.free(a_buf)
    memory_pool.free(b_buf)
    memory_pool.free(c_buf)

    result = Tensor(
        result_data, requires_grad=tensor_a.requires_grad or tensor_b.requires_grad
    )

    return result


class Operation:
    def __init__(self):
        self.tensors = []
        self.grad_fn = None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class AddOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, add_program.add_kernel)
        result.creator = self
        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            a.backward(grad_output)  # Gradient of addition wrt a is 1
        if b.requires_grad:
            b.backward(grad_output)  # Gradient of addition wrt b is 1


class MultiplyOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, multiply_program.multiply_kernel)
        result.creator = self
        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            # Gradient of a with respect to d = a * b is b
            a_grad = grad_output * b.data
            a.backward(a_grad)
        if b.requires_grad:
            # Gradient of b with respect to d = a * b is a
            b_grad = grad_output * a.data
            b.backward(b_grad)


class SubtractOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, subtract_program.subtract_kernel)
        result.creator = self
        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            # Gradient of a with respect to c = a - b is 1
            a.backward(grad_output)
        if b.requires_grad:
            # Gradient of b with respect to c = a - b is -1
            b.backward(-grad_output)


class MatmulOperation(Operation):
    def forward(self, a, b):
        if a.data.ndim != 2 or b.data.ndim != 2:
            raise ValueError(
                "Matmul operation requires both tensors to be 2D matrices."
            )
        if a.data.shape[1] != b.data.shape[0]:
            raise ValueError(
                "Matrix multiplication requires shape alignment: a's columns must match b's rows."
            )
        self.tensors = [a, b]
        result = tensor_operation(a, b, matmul_program.matmul_kernel, is_matmul=True)
        result.creator = self
        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            # Compute grad_a using numpy's matmul directly
            grad_a = np.matmul(grad_output, b.data.T)
            a.backward(grad_a)
        if b.requires_grad:
            # Compute grad_b using numpy's matmul directly
            grad_b = np.matmul(a.data.T, grad_output)
            b.backward(grad_b)


def add(a, b):
    op = AddOperation()
    return op.forward(a, b)


def multiply(a, b):
    op = MultiplyOperation()
    return op.forward(a, b)


def subtract(a, b):
    op = SubtractOperation()
    return op.forward(a, b)


def matmul(a, b):
    op = MatmulOperation()
    return op.forward(a, b)
