import matplotlib.pyplot as plt
import numpy as np

A = np.array([[1, 2, 3], [3, 2, 3]])
B = np.array([3, 2, 1])

print(A.shape)
print(B.shape)

print(B <= A)
print((B <= A).shape)