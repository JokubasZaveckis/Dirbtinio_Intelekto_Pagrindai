import numpy as np
from neuron import ArtificialNeuron
from generate_data import X, y  

np.random.seed(42)

def find_threshold_weights():
    valid_weights = []
    max_attempts = 10000  
    attempt = 0

    while len(valid_weights) < 3 and attempt < max_attempts:
        attempt += 1
        w1 = np.random.uniform(-5, 5)
        w2 = np.random.uniform(-5, 5)
        b = np.random.uniform(-5, 5)

        neuron = ArtificialNeuron(w1, w2, b, activation_function="threshold")

        # Get predictions
        predictions = [neuron.compute_output(x1, x2) for x1, x2 in X]
        accuracy = np.mean(np.array(predictions) == y) * 100  # Skaiciuoti tiksluma

        if accuracy == 100:  # Saugoti tik toblus svorius
            valid_weights.append((w1, w2, b))
            print(f"Rastas tinkamas rinkinys {len(valid_weights)}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}, Tikslumas={accuracy:.2f}%")

    if len(valid_weights) < 3:
        print("Nepavyko rasti 3 tinkamų svorių rinkinių")

    return valid_weights

if __name__ == "__main__":
    valid_sets = find_threshold_weights()
    print("\nGalutiniai tinkami svorių rinkiniai (slenkstinė funkcija):")
    for i, (w1, w2, b) in enumerate(valid_sets, start=1):
        print(f"Set {i}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
