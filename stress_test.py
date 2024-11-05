import numpy as np
import time

from rapidautograd.tensor import Tensor, add, subtract, multiply, matmul

def stress_test_tensor_operations():
    # Define the tensor size for a stress test
    large_size = 2048  # This creates 2048x2048 elements for matmul, adjust as needed based on memory

    # Generate large random data for the tensors
    a_data = np.random.rand(large_size, large_size).astype(np.float32)
    b_data = np.random.rand(large_size, large_size).astype(np.float32)

    # Flatten the data to fit into the Tensor class as a 1D array
    a = Tensor(a_data.flatten(), requires_grad=True)
    b = Tensor(b_data.flatten(), requires_grad=True)

    # Stress test - Time multiple operations
    num_iterations = 10
    print(f"Stress testing with {num_iterations} iterations on tensors of size {large_size}x{large_size}...")

    # Measure time for each operation type
    start_time = time.time()

    # Addition test
    for _ in range(num_iterations):
        added = add(a, b)
    print("Addition time:", time.time() - start_time)

    # Subtraction test
    start_time = time.time()
    for _ in range(num_iterations):
        subtracted = subtract(a, b)
    print("Subtraction time:", time.time() - start_time)

    # Element-wise multiplication test
    start_time = time.time()
    for _ in range(num_iterations):
        multiplied = multiply(a, b)
    print("Element-wise multiplication time:", time.time() - start_time)

    # Matrix multiplication test (expect this to be the most intensive)
    start_time = time.time()
    for _ in range(num_iterations):
        matmul_result = matmul(a, b)
    print("Matrix multiplication time:", time.time() - start_time)

    # Chain operations: ((a + b) - a) * b
    start_time = time.time()
    for _ in range(num_iterations):
        chain_result = multiply(subtract(add(a, b), a), b)
    print("Chained operations time ((a + b) - a) * b:", time.time() - start_time)

    print("Stress test completed successfully.")

if __name__ == "__main__":
    stress_test_tensor_operations()