import numpy as np
from rapidautograd.tensor import Tensor, add, multiply, subtract, matmul


def test_addition():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = add(a, b)
    c.backward(np.array([1.0, 1.0, 1.0, 0.0]))  # Assuming padding to multiple of 4
    expected_grad_a = [1.0, 1.0, 1.0, 0.0]
    assert np.allclose(a.grad, expected_grad_a), "Addition test failed for tensor a"
    print("Addition test passed.")


def test_multiplication():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    d = multiply(a, b)
    d.backward(np.array([1.0, 1.0, 1.0, 0.0]))  # Assuming padding to multiple of 4
    expected_grad_a = [4.0, 5.0, 6.0, 0.0]
    assert np.allclose(
        a.grad, expected_grad_a
    ), "Multiplication test failed for tensor a"
    print("Multiplication test passed.")


def test_subtraction():
    a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    e = subtract(a, b)
    e.backward(np.array([1.0, 1.0, 1.0, 0.0]))  # Assuming padding to multiple of 4
    expected_grad_a = [1.0, 1.0, 1.0, 0.0]
    expected_grad_b = [-1.0, -1.0, -1.0, 0.0]
    assert np.allclose(a.grad, expected_grad_a), "Subtraction test failed for tensor a"
    assert np.allclose(b.grad, expected_grad_b), "Subtraction test failed for tensor b"
    print("Subtraction test passed.")


def test_matrix_multiplication():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    f = matmul(a, b)
    # Correct gradient dimensions for testing matmul
    f.backward(
        np.array([[1.0, 1.0], [1.0, 1.0]])
    )  # Correct dimensions for a 2x2 result

    # Expected gradients:
    # grad_a = grad_output @ b.transpose() = [[1,1],[1,1]] @ [[2,0],[0,2]] = [[2,2],[2,2]]
    expected_grad_a = [[2.0, 2.0], [2.0, 2.0]]
    # grad_b = a.transpose() @ grad_output = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    expected_grad_b = [[4.0, 4.0], [6.0, 6.0]]

    assert np.allclose(
        a.grad, expected_grad_a
    ), "Matrix multiplication test failed for tensor a"
    assert np.allclose(
        b.grad, expected_grad_b
    ), "Matrix multiplication test failed for tensor b"
    print("Matrix multiplication test passed.")


def test_complex_operations():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = add(a, b)  # c = a + b
    d = multiply(a, c)  # d = a * c
    d.backward(np.array([1.0, 1.0, 1.0, 0.0]))  # Assuming padding to multiple of 4

    # Expected gradients:
    # a.grad = grad_d * c + grad_c * a = [1,1,1,0] * [5,7,9,0] + [1,2,3,0] * [1,1,1,0] = [5+1,7+2,9+3,0+0] = [6,9,12,0]
    expected_grad_a = [6.0, 9.0, 12.0, 0.0]
    # b.grad = grad_c * 1 = [1,2,3,0] * 1 = [1,2,3,0]
    expected_grad_b = [1.0, 2.0, 3.0, 0.0]

    assert np.allclose(
        a.grad, expected_grad_a
    ), "Complex operation test failed for tensor a"
    assert np.allclose(
        b.grad, expected_grad_b
    ), "Complex operation test failed for tensor b"
    print("Complex operation test passed.")


def run_tests():
    test_addition()
    test_multiplication()
    test_subtraction()
    test_matrix_multiplication()
    test_complex_operations()


if __name__ == "__main__":
    run_tests()
