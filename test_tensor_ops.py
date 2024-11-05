import numpy as np
from rapidautograd.tensor import Tensor, add, subtract, matmul, multiply

def test_tensor_operations():
    # Create test tensors
    a_data = np.array([1.0, 2.0, 3.0, 4.0])
    b_data = np.array([4.0, 3.0, 2.0, 1.0])

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    # Test addition
    added = add(a, b)
    print("Addition Result:", added)
    assert np.allclose(added.data, a_data + b_data), "Addition operation failed"

    # Test subtraction
    subtracted = subtract(a, b)
    print("Subtraction Result:", subtracted)
    assert np.allclose(subtracted.data, a_data - b_data), "Subtraction operation failed"

    # Prepare data for matrix multiplication
    # Using reshaped data to form a 2x2 matrix
    c_data = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
    d_data = np.array([4.0, 3.0, 2.0, 1.0]).reshape(2, 2)
    c = Tensor(c_data.flatten(), requires_grad=True)
    d = Tensor(d_data.flatten(), requires_grad=True)

    # Test matrix multiplication
    matmul_result = matmul(c, d)
    expected_matmul_result = np.matmul(c_data, d_data).flatten()
    print("Matrix Multiplication Result:", matmul_result)
    assert np.allclose(matmul_result.data, expected_matmul_result), "Matrix multiplication operation failed"

if __name__ == "__main__":
    test_tensor_operations()
