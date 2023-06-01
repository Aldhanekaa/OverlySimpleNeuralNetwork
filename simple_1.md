PLEASE RUN

# Very Simple Neural Network 1

this code (`simple_1.py`) is a program to train a simple neural network.

this neural network doesnt have hidden layer, so it only has input and output layer, yeah thats why this is a very simple neural network.

And also this neural network takes 3 neurons as input and 1 neuron as outputs, and more interestingly it only has 3 weights.

## Data

```math
inputs = \begin{bmatrix}
            0 & 0 & 1 \\
            1 & 1 & 1 \\
            1 & 0 & 1 \\
            0 & 1 & 1

        \end{bmatrix}$$
```

```math
outputs = \begin{bmatrix}
        0 \\ 1 \\ 1 \\ 0
        \end{bmatrix}

```

imagine that input as a table, and you'll put the output beside that table

```math
\begin{bmatrix}
            0 & 0 & 1 \\
            1 & 1 & 1 \\
            1 & 0 & 1 \\
            0 & 1 & 1

        \end{bmatrix} = \begin{bmatrix}
        0 \\ 1 \\ 1 \\ 0
        \end{bmatrix}
```

## Forward Propagation

the inputs consist of `4x3` matrix, whereas each row is the input
the output consist of `4x1` matrix, whereach each row is the output for each row in training inputs.

the weights itself generated randomly as the code below

```python
sypnatic_weights = 2 * np.random.random((3,1)) - 1
```

In order to calculate the output neuron, we need to have the sum of 3 inputs and 3 weights multiplication :

| Programmatically         |                         Mathematically                         |
| :----------------------- | :------------------------------------------------------------: |
| $$input \times weights$$ |                 $$Z = \sum\_{i=1}^{3}x_iw_i$$                  |
| s                        | $x_i = \text{input neurons}$ <br/> $w_i = \text{weight layer}$ |

<details open>
<summary>Detailed Formula</summary>
<br>

```math
    Z =\begin{bmatrix}
            0 & 0 & 1 \\
            1 & 1 & 1 \\
            1 & 0 & 1 \\
            0 & 1 & 1

        \end{bmatrix}

        \times

        \begin{bmatrix}
        -0.16595599 \\ 0.44064899 \\ -0.99977125
        \end{bmatrix}

        =

        \begin{bmatrix}
        0 \times -0.16595599 + 0 \times 0.44064899 + 1 \times -0.99977125 \\
        1 \times -0.16595599 + 1 \times 0.44064899 + 1 \times -0.99977125 \\
        1 \times -0.16595599 + 0 \times 0.44064899 + 1 \times -0.99977125 \\
        0 \times -0.16595599 + 1 \times 0.44064899 + 1 \times -0.99977125
        \end{bmatrix}
```

or the formula can be simplify to

```math
\Phi(x) = 1/{1 + e^{-\sum_{i=1}^{3}x_iw_i }}
```

</details>

---

### Activation Function

Then we would calculate the activation function using sigmoid $ \Phi $

```math
\Phi(Z) = 1 / 1 + e^{-Z}
```

and that formula describes as the code in function sigmoid in line 3:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## Backward Propagation

```math
error \times input \times \Phi'(output)
```

## Testing the NN

To put the NN into test, just run the forward propagation

```
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
```

documented by Aldhanekaa.
