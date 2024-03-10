import numpy as np


class MLP(object):
    def __init__(self, num_inputs=3, num_hiddens=[3, 4], num_outputs=2) -> None:

        # def variables
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        # layers structure
        layers = [self.num_inputs] + self.num_hiddens + [self.num_outputs]
        print(f"Number of layers {layers}")

        # random weights
        self.weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            print(f"{i} is {w}")
            self.weights.append(w)
        print(f"Number of weight {self.weights}")

        # forward propagation

    def forward_propagate(self, inputs):

        activations = inputs

        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate the activations         
            activations = self._sigmoid(net_inputs)
            print(f"Number of activations {activations}")

        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create a mlp instance
    mlp = MLP()
    # create an inputs
    inputs = np.random.rand(mlp.num_inputs)
    # perform forward prop
    outputs = mlp.forward_propagate(inputs)
    # print output
    print("The network input is : {}".format(inputs))
    print("The network output is: {}".format(outputs))
