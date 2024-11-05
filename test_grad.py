from rapidautograd.tensor import Tensor, add, multiply

# Create tensors
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
c = add(a, b)          # c = a + b
d = multiply(a, c)     # d = a * c

# Compute gradients
d.backward()

# Output gradients
print("Gradient of a:", a.grad)
print("Gradient of b:", b.grad)
