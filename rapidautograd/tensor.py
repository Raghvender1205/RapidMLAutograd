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
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __mul__(self, other):
        return multiply(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

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
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return max(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return min(self, axis=axis, keepdims=keepdims)

###############
# TensorOps
##############

class Operation:
    """
    Base Operation
    """
    def __init__(self):
        self.tensors = []
        self.grad_fn = None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

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
        # Prepare data for OpenCL
        data_a = tensor_a.data.astype(np.float32)
        data_b = tensor_b.data.astype(np.float32)
        result_shape = (tensor_a.data.shape[0], tensor_b.data.shape[1])
        result_data = np.empty(result_shape, dtype=np.float32)

        # Create buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_a
        )
        b_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_b
        )
        c_buf = cl.Buffer(context, mf.WRITE_ONLY, result_data.nbytes)

        # Set kernel arguments and execute
        N = np.int32(tensor_a.data.shape[0])
        M = np.int32(tensor_b.data.shape[1])
        K = np.int32(tensor_a.data.shape[1])  # same as tensor_b.data.shape[0]

        matmul_kernel = operation_kernel.matmul_kernel
        matmul_kernel.set_args(a_buf, b_buf, c_buf, N, M, K)
        global_size = (N, M)
        cl.enqueue_nd_range_kernel(queue, matmul_kernel, global_size, None)

        # Read the result
        cl.enqueue_copy(queue, result_data, c_buf).wait()

        # Release buffers
        a_buf.release()
        b_buf.release()
        c_buf.release()

        result = Tensor(
            result_data,
            requires_grad=tensor_a.requires_grad or tensor_b.requires_grad,
        )
        return result
    else:
        # Compute broadcasted shape
        broadcasted_shape = np.broadcast_shapes(tensor_a.shape, tensor_b.shape)
        # Broadcast data to the same shape
        data_a = np.broadcast_to(tensor_a.data, broadcasted_shape).astype(
            np.float32
        )
        data_b = np.broadcast_to(tensor_b.data, broadcasted_shape).astype(
            np.float32
        )
        # Flatten the data for OpenCL
        data_a_flat = data_a.flatten()
        data_b_flat = data_b.flatten()
        result_data_flat = np.empty_like(data_a_flat)

        # Create buffers
        mf = cl.mem_flags
        a_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_a_flat
        )
        b_buf = cl.Buffer(
            context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_b_flat
        )
        c_buf = cl.Buffer(context, mf.WRITE_ONLY, data_a_flat.nbytes)

        # Execute the kernel
        global_size = (data_a_flat.size,)
        operation_kernel(queue, global_size, None, a_buf, b_buf, c_buf, np.int32(data_a_flat.size))

        # Read the result
        cl.enqueue_copy(queue, result_data_flat, c_buf).wait()
        result_data = result_data_flat.reshape(broadcasted_shape)

        # Release buffers
        a_buf.release()
        b_buf.release()
        c_buf.release()

        result = Tensor(
            result_data,
            requires_grad=tensor_a.requires_grad or tensor_b.requires_grad,
        )

        return result


def unbroadcast(grad, shape):
    # Sum grad over axes where shape is 1 (broadcasted dimensions)
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class AddOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, add_program.add_kernel)
        result.creator = self
        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            grad_a = unbroadcast(grad_output, a.shape)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output, b.shape)
            b.backward(grad_b)


class MultiplyOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, multiply_program.multiply_kernel)
        result.creator = self

        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            grad_a = grad_output * b.data
            grad_a = unbroadcast(grad_a, a.shape)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = grad_output * a.data
            grad_b = unbroadcast(grad_b, b.shape)
            b.backward(grad_b)


class SubtractOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, subtract_program.subtract_kernel)
        result.creator = self

        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            grad_a = unbroadcast(grad_output, a.shape)
            a.backward(grad_a)
        if b.requires_grad:
            grad_b = unbroadcast(-grad_output, b.shape)
            b.backward(grad_b)


class MatmulOperation(Operation):
    def forward(self, a, b):
        self.tensors = [a, b]
        result = tensor_operation(a, b, matmul_program, is_matmul=True)
        result.creator = self

        return result

    def backward(self, grad_output):
        a, b = self.tensors
        if a.requires_grad:
            grad_a = np.matmul(grad_output, b.data.T)
            a.backward(grad_a)
        if b.requires_grad:
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

#############
# ReduceOps
#############

class SumOperation(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.tensors = [a]
        result_data = a.data.sum(axis=self.axis, keepdims=self.keepdims)
        result = Tensor(result_data, requires_grad=a.requires_grad)
        result.creator = self

        return result

    def backward(self, grad_output):
        a = self.tensors[0]
        if a.requires_grad:
            grad_input = grad_output
            if self.axis is not None:
                shape = list(a.shape)
                if not self.keepdims:
                    for ax in sorted((self.axis,) if isinstance(self.axis, int) else self.axis):
                        shape[ax] = 1
                grad_input = grad_output.reshape(shape)
            grad_input = np.broadcast_to(grad_input, a.shape)
            a.backward(grad_input)

class MeanOperation(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.tensors = [a]
        self.input_shape = a.shape
        result_data = a.data.mean(axis=self.axis, keepdims=self.keepdims)
        self.output_shape = result_data.shape
        result = Tensor(result_data, requires_grad=a.requires_grad)
        result.creator = self

        return result

    def backward(self, grad_output):
        a = self.tensors[0]
        if a.requires_grad:
            grad_input = grad_output
            if self.axis is not None:
                shape = list(a.shape)
                if not self.keepdims:
                    for ax in sorted((self.axis,) if isinstance(self.axis, int) else self.axis):
                        shape[ax] = 1
                grad_input = grad_output.reshape(shape)
            grad_input = np.broadcast_to(grad_input, a.shape) / np.prod(
                [a.shape[ax] for ax in (self.axis,) if self.axis is not None])
            a.backward(grad_input)

class MaxOperation(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.tensors = [a]
        self.a_data = a.data
        self.result_data = a.data.max(axis=self.axis, keepdims=self.keepdims)
        result = Tensor(self.result_data, requires_grad=a.requires_grad)
        result.creator = self
        return result

    def backward(self, grad_output):
        a = self.tensors[0]
        if a.requires_grad:
            grad_input = np.zeros_like(a.data)
            mask = (a.data == self.result_data)
            grad = grad_output
            if self.axis is not None and not self.keepdims:
                grad = np.expand_dims(grad, axis=self.axis)
            grad_input[mask] = grad
            a.backward(grad_input)

class MinOperation(Operation):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, a):
        self.tensors = [a]
        self.a_data = a.data
        self.result_data = a.data.min(axis=self.axis, keepdims=self.keepdims)
        result = Tensor(self.result_data, requires_grad=a.requires_grad)
        result.creator = self
        return result

    def backward(self, grad_output):
        a = self.tensors[0]
        if a.requires_grad:
            grad_input = np.zeros_like(a.data)
            mask = (a.data == self.result_data)
            grad = grad_output
            if self.axis is not None and not self.keepdims:
                grad = np.expand_dims(grad, axis=self.axis)
            grad_input[mask] = grad
            a.backward(grad_input)


def sum(tensor: Tensor, axis=None, keepdims=True):
    op = SumOperation(axis=axis, keepdims=keepdims)

    return op.forward(tensor)

def mean(tensor: Tensor, axis=None, keepdims=True):
    op = MeanOperation(axis=axis, keepdims=keepdims)

    return op.forward(tensor)

def max(tensor: Tensor, axis=None, keepdims=True):
    op = MaxOperation(axis=axis, keepdims=keepdims)

    return op.forward(tensor)

def min(tensor: Tensor, axis=None, keepdims=True):
    op = MinOperation(axis=axis, keepdims=keepdims)

    return op.forward(tensor)


