import numpy as np
import matplotlib.pyplot as plt
from generate_data import X, y 
from sigmoid_activation import find_sigmoid_weights 

# Gauna tinkamus svoriu rinkinius
valid_weights = find_sigmoid_weights()

# Nubraizo sugeneruotus duomenu taskus
plt.figure(figsize=(6, 6))

# Plot the generated data points
plt.scatter(X[:10, 0], X[:10, 1], color='blue', label='Klasė 0')
plt.scatter(X[10:, 0], X[10:, 1], color='red', label='Klasė 1')

# Funkcija sprendimo ribos braizymui
def plot_decision_boundary(w1, w2, b, color, label):
    x_vals = np.linspace(0, 5, 100)
    y_vals = -(w1 * x_vals + b) / w2  
    plt.plot(x_vals, y_vals, color=color, label=label)

# Funkcija nubrezti vektoriu
def plot_weight_vector(w1, w2, b, color):
    # Pick a reference point on the line
    x_ref = 2
    y_ref = -(w1 * x_ref + b) / w2
    
    magnitude = np.sqrt(w1**2 + w2**2)
    
    # Sumaiznti vektoriu
    scale_factor = 2.0 / magnitude
    
    w1_scaled = w1 * scale_factor
    w2_scaled = w2 * scale_factor
    
    # Nupiesia vektoriu
    plt.quiver(
        x_ref, y_ref, w1_scaled, w2_scaled,
        color=color, angles='xy', scale_units='xy',
        scale=1, width=0.01
    )

# Braizo tris sprendimo ribas ir ju svorio vektorius
colors = ['green', 'purple', 'orange']
for i, (w1, w2, b) in enumerate(valid_weights):
    plot_decision_boundary(w1, w2, b, colors[i], f"Riba {i+1}")
    plot_weight_vector(w1, w2, b, colors[i])

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sprendimo ribos su jų vektoriais")
plt.legend()
plt.grid(True)


plt.xlim(0, 5)
plt.ylim(0, 5)


plt.gca().set_aspect('equal', adjustable='box')

plt.show()
