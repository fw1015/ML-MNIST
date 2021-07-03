import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class neuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # m (Input Layer)
        self.input_nodes = input_nodes
        # m (Hidden Layer)
        self.hidden_nodes = hidden_nodes
        # m (Output Layer)
        self.output_nodes = output_nodes
        # W12
        self.weight1 = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # W23
        self.weight2 = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        # alpha
        self.learning_rate = learning_rate
        # Sigmoid(Z)
        self.activation = lambda x:scipy.special.expit(x)
        pass

    # Training Sets
    def train(self, inputs_list, targets_list):
        # Modelling Testing & Training data
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # Z1
        hidden_inputs = np.dot(self.weight1, inputs)
        # A1
        hidden_outputs = self.activation(hidden_inputs)
        # Z2
        final_inputs = np.dot(self.weight2, hidden_outputs)
        # A2
        final_outputs = self.activation(final_inputs)
        # delta (Output Layer)
        actual_errors = targets - final_outputs
        # delta (Hidden Layer)
        hidden_errors = np.dot(self.weight2.T, actual_errors)
        # delta deviation
        self.weight2 += self.learning_rate * np.dot((actual_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weight1 += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # Testing Sets
    def query(self, inputs_list):
        # Modelling inputs list
        inputs = np.array(inputs_list, ndmin=2).T
        # Z1
        hidden_inputs = np.dot(self.weight1, inputs)
        # A1
        hidden_outputs = self.activation(hidden_inputs)
        # Z2
        final_inputs = np.dot(self.weight2, hidden_outputs)
        # A2
        final_outputs = self.activation(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
alpha = 0.3
ml = neuralNetwork(input_nodes, hidden_nodes, output_nodes, alpha)

training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for sample in training_data_list:
    all_values = sample.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    ml.train(inputs, targets)
    pass

test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(ml.query(inputs))