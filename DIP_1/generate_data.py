import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generuoja 2 klases po 10 tasku
num_points = 10

# Klase 0 (kaire apatine koordinaciu plokstumos dalis)
class_0_x1 = np.random.uniform(0, 2, num_points)
class_0_x2 = np.random.uniform(0, 2, num_points)
class_0 = np.column_stack((class_0_x1, class_0_x2))

# Klase 1 (Desine virsutine koordinaciu plokstumos dalis)
class_1_x1 = np.random.uniform(3, 5, num_points)
class_1_x2 = np.random.uniform(3, 5, num_points)
class_1 = np.column_stack((class_1_x1, class_1_x2))

# Sujungia klases i viena bendra duomenu masyva
X = np.vstack((class_0, class_1))
y = np.hstack((np.zeros(num_points), np.ones(num_points)))

print("Sugeneruotų duomenų taškai:")
for i in range(len(X)):
    print(f"Point {i+1}: (x1={X[i,0]:.2f}, x2={X[i,1]:.2f}) -> Class {int(y[i])}")

# Nubraizo sugeneruotus duomenis
plt.figure(figsize=(6,6))
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Klasė 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Klasė 1')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sugeneruoti duomenys")
plt.legend()
plt.grid(True)
plt.show()
