# test_tensor_ops.py

import numpy as np
from rapidautograd.tensor import Tensor, add, multiply, subtract, matmul

def test_addition():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = add(a, b)
    c.backward(np.array([1.0, 1.0, 1.0]))
    expected_grad_a = [1.0, 1.0, 1.0]
    assert np.allclose(a.grad, expected_grad_a), "Addition test failed for tensor a"
    print("Addition test passed.")

def test_multiplication():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    d = multiply(a, b)
    d.backward(np.array([1.0, 1.0, 1.0]))
    expected_grad_a = [4.0, 5.0, 6.0]
    expected_grad_b = [1.0, 2.0, 3.0]
    assert np.allclose(a.grad, expected_grad_a), "Multiplication test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Multiplication test failed for tensor b"
    print("Multiplication test passed.")

def test_subtraction():
    a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    e = subtract(a, b)
    e.backward(np.array([1.0, 1.0, 1.0]))
    expected_grad_a = [1.0, 1.0, 1.0]
    expected_grad_b = [-1.0, -1.0, -1.0]
    assert np.allclose(a.grad, expected_grad_a), "Subtraction test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Subtraction test failed for tensor b"
    print("Subtraction test passed.")

def test_matrix_multiplication():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    f = matmul(a, b)
    f.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))
    expected_grad_a = [[2.0, 2.0], [2.0, 2.0]]
    expected_grad_b = [[4.0, 4.0], [6.0, 6.0]]
    assert np.allclose(a.grad, expected_grad_a), "Matrix multiplication test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Matrix multiplication test failed for tensor b"
    print("Matrix multiplication test passed.")

def test_complex_operations():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = add(a, b)  # c = a + b
    d = multiply(a, c)  # d = a * c
    d.backward(np.array([1.0, 1.0, 1.0]))

    # Expected gradients:
    # For a: grad = [6.0, 9.0, 12.0]
    expected_grad_a = [6.0, 9.0, 12.0]
    # For b: grad = [1.0, 2.0, 3.0]
    expected_grad_b = [1.0, 2.0, 3.0]

    assert np.allclose(a.grad, expected_grad_a), "Complex operation test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Complex operation test failed for tensor b"
    print("Complex operation test passed.")


def test_broadcasting_addition():
    a = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)  # Shape (3,1)
    b = Tensor([4.0, 5.0], requires_grad=True)             # Shape (2,)
    c = add(a, b)                                          # Result shape (3,2)
    c.backward(np.ones_like(c.data))

    # Expected gradients for a and b
    # For a: Sum over the broadcasted dimension (axis=1)
    expected_grad_a = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).sum(axis=1, keepdims=True)
    # For b: Sum over the broadcasted dimension (axis=0)
    expected_grad_b = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).sum(axis=0)

    assert np.allclose(a.grad, expected_grad_a), "Broadcasting addition test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Broadcasting addition test failed for tensor b"
    print("Broadcasting addition test passed.")

def test_broadcasting_multiplication():
    a = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)  # Shape (3,1)
    b = Tensor([4.0, 5.0], requires_grad=True)             # Shape (2,)
    c = multiply(a, b)                                     # Result shape (3,2)
    c.backward(np.ones_like(c.data))

    # Expected gradients for a and b
    expected_grad_a = b.data.sum(axis=0).reshape(1, -1) * np.ones((3, 1))
    expected_grad_a = expected_grad_a.sum(axis=1, keepdims=True)
    expected_grad_b = a.data.sum(axis=0).reshape(-1, 1).T * np.ones((1, 2))
    expected_grad_b = expected_grad_b.sum(axis=0)

    assert np.allclose(a.grad, expected_grad_a), "Broadcasting multiplication test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Broadcasting multiplication test failed for tensor b"
    print("Broadcasting multiplication test passed.")

def run_tests():
    test_addition()
    test_multiplication()
    test_subtraction()
    test_matrix_multiplication()
    test_complex_operations()
    test_broadcasting_addition()
    test_broadcasting_multiplication()

if __name__ == "__main__":
    run_tests()
