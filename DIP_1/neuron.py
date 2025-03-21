import numpy as np

class ArtificialNeuron:
    def __init__(self, w1, w2, bias, activation_function="threshold"): # Inicijuoja neurono svorius, poslinki ir aktyvacijos funkcija
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.activation_function = activation_function

    def activate(self, a): # Pritaiko pasirinkitos aktyvacijos funkcija
        if self.activation_function == "threshold":
            return 1 if a >= 0 else 0
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-a))  # Sigmoidine funkcija

    def compute_output(self, x1, x2): # Apskaiciuoja neurono isvesti pagal svorius ir iejimus
        a = (self.w1 * x1) + (self.w2 * x2) + self.bias
        return self.activate(a)

# Testavimas su pavyzdiniais iejimais
if __name__ == "__main__":
    neuron = ArtificialNeuron(w1=1.0, w2=1.0, bias=-2.0, activation_function="threshold")

    # Test the neuron with sample inputs
    test_inputs = [(0, 0), (1, 1), (2, 2), (3, 3)]
    
    print("Neuron Output:")
    for x1, x2 in test_inputs:
        output = neuron.compute_output(x1, x2)
        print(f"Input: (x1={x1}, x2={x2}) -> Output: {output}")
