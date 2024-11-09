import numpy as np
from rapidautograd.tensor import Tensor

a = Tensor(np.random.randn(4, 4), requires_grad=True)
b = Tensor(np.random.randn(4, 4), requires_grad=True)
s = a.add(b)
s.backward()

print(a)
print(s)
print(s.grad)