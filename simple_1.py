import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


training_inputs = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

training_outputs = np.array([[0,1,1,0]]).T
 
print(training_outputs.shape)
np.random.seed(1)

# one layer of neural network, three rows
sypnatic_weights = 2 * np.random.random((3,1)) - 1


print("Training Inputs")
print(training_inputs)

print('\nRandom starting sypnatic weights: ')
print(sypnatic_weights)

# print(np.dot(training_inputs, sypnatic_weights))
for iteration in range(100000):

    input_layer = training_inputs

    z = np.dot(input_layer, sypnatic_weights) # this is the z
    outputs = sigmoid(z) # the activation each neuron in output

    error = training_outputs - outputs # cost function

    adjustments = error * sigmoid_derivative(outputs) # cost * gradient
    
    sypnatic_weights += np.dot(input_layer.T, adjustments)



print("Outputs after training: ")
print(outputs)

inputs = np.zeros(3)
j = 0
while j < 3:
    print(f"Input for {j}: ")
    a = input("> ")
    a = int(a)
    inputs[j] = a
    j += 1
print("inputs", inputs)

inputs = inputs.T
z = np.dot(inputs, sypnatic_weights) 
# print("Z ", z)

out = sigmoid(z)
print("Computer has guessed it!", out)