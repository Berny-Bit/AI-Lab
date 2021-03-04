# NeuralNetwork from scratch
import numpy as np


# activation function
def relu(value):
    output = max(value, 0)
    return output


input_data = np.array([2, 3])

# dictionary with weights
weights = {'node_0': np.array([1, 1]),
           'node_1': np.array([-1, 1]),
           'output': np.array([2, -1])}

# forward propagation and activation function
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# put values of hidden layer in array for further computation
hidden_layer_values = np.array([node_0_output, node_1_output])

# forward propagation
output = (hidden_layer_values * weights['output']).sum()

if __name__ == '__main__':
    print("output: ", output)
