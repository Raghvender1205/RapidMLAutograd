import numpy as np
import time

from rapidautograd.tensor import Tensor, add, subtract, multiply, matmul


def stress_test_tensor_operations():
    # Define the tensor size for a stress test
    large_size = 2048  # This creates 2048x2048 matrices for matmul

    # Generate large random data for element-wise operations
    # Ensure that the total number of elements is a multiple of 4 to avoid unnecessary padding
    a_data_1d = np.random.rand(large_size * large_size).astype(np.float32)
    b_data_1d = np.random.rand(large_size * large_size).astype(np.float32)

    # Generate large random data for matrix multiplication
    a_data_2d = np.random.rand(large_size, large_size).astype(np.float32)
    b_data_2d = np.random.rand(large_size, large_size).astype(np.float32)

    # Create 1D Tensors for element-wise operations
    a1d = Tensor(a_data_1d, requires_grad=True)
    b1d = Tensor(b_data_1d, requires_grad=True)

    # Create 2D Tensors for matrix multiplication
    a2d = Tensor(a_data_2d, requires_grad=True)
    b2d = Tensor(b_data_2d, requires_grad=True)

    # Number of iterations for stress testing
    num_iterations = 10
    print(
        f"Stress testing with {num_iterations} iterations on tensors of size {large_size}x{large_size}...\n"
    )

    # Measure time for addition (element-wise)
    start_time = time.time()
    for _ in range(num_iterations):
        added = add(a1d, b1d)  # Element-wise addition
    addition_time = time.time() - start_time
    print(f"Addition time for 1D tensors: {addition_time:.4f} seconds")

    # Measure time for subtraction (element-wise)
    start_time = time.time()
    for _ in range(num_iterations):
        subtracted = subtract(a1d, b1d)  # Element-wise subtraction
    subtraction_time = time.time() - start_time
    print(f"Subtraction time for 1D tensors: {subtraction_time:.4f} seconds")

    # Measure time for element-wise multiplication
    start_time = time.time()
    for _ in range(num_iterations):
        multiplied = multiply(a1d, b1d)  # Element-wise multiplication
    multiply_time = time.time() - start_time
    print(
        f"Element-wise multiplication time for 1D tensors: {multiply_time:.4f} seconds"
    )

    # Measure time for matrix multiplication
    start_time = time.time()
    for _ in range(num_iterations):
        matmul_result = matmul(a2d, b2d)  # Matrix multiplication
    matmul_time = time.time() - start_time
    print(f"Matrix multiplication time for 2D tensors: {matmul_time:.4f} seconds")

    # Measure time for chain operations: ((a + b) - a) * b (element-wise)
    start_time = time.time()
    for _ in range(num_iterations):
        chain_result = multiply(subtract(add(a1d, b1d), a1d), b1d)  # ((a + b) - a) * b
    chain_time = time.time() - start_time
    print(
        f"Chained operations time ((a + b) - a) * b for 1D tensors: {chain_time:.4f} seconds"
    )

    print("\nStress test completed successfully.")


if __name__ == "__main__":
    stress_test_tensor_operations()
