import numpy as np

A = np.array([[1, 4, -1], [2, 0, 1]])
B = np.array([[-1, 0], [1, 3], [-1, 1]])

#A vuông & detA != 0 => khả nghịch
A_inv = np.linalg.pinv(A)
print(A_inv)

print("A = \n", A)
print("\nB = \n", B)

print("\nA + B^T = \n", A + B.T)

print("\nA - B^T = \n", A - B.T)

print("\nA * 2 = \n", A*2)

print("\nA * B = \n", A @ B)

print("\nA * A^-1 = \n", A @ A_inv)
