# A Simple Multilayer Perceptron


Using the library, it is fairly easy to build a simple MLP.

First we import the libraries:

```python
import numpy as np
import revgraph.core as rc
```

Then we define the mappings that the perceptron's going to classify. 
(one-hot encoded xor function):

```python
# Training Data
x = [[0,0],
     [0,1],
     [1,0],
     [1,1]]

y = [[0,1],
     [1,0],
     [1,0],
     [0,1]]
```

Then we specify the architecture of our perceptron:

```python
# Multilayer Perceptron consists of:
#     2 input units (input x)
#     2 hidden units
#     2 output units (output y)
in_units, hidden_units, out_units = 2, 8, 2
```

Define the custom placeholders for feeding in training data.

```python
x_ph = rc.placeholder(name='x', shape=(None, in_units))
y_ph = rc.placeholder(name='y', shape=(None, out_units))
```

Then we define the weights and bias in the network:
```python
# Define the weights and bias
w0 = rc.variable(np.random.randn(in_units, hidden_units))
b0 = rc.variable(np.random.randn(1, hidden_units))
w1 = rc.variable(np.random.randn(hidden_units, out_units))
b1 = rc.variable(np.random.randn(1, out_units))
```

Let's use the sigmoid function as our activation function:

```python
def sigmoid(z: rc.tensor) -> rc.tensor:
    return 1 / (1 + rc.exp(-z))
```

We use MSE as a metric for how good the perceptron is performing:

```python
def mse(y_true: rc.tensor, y_pred: rc.tensor) -> rc.tensor:
    return ((y_true - y_pred) ** 2) / rc.len(y_true)
```

And then the main classifier itself: the objective is to minimize the 
mean sqaured error:

```python
l0 = x_ph
l1 = rc.tanh(l0.dot(w0) + b0)
predict = sigmoid(l1.dot(w1) + b1)
loss = mse(y_ph, predict)
fit = rc.sgd(lr=0.1, momentum=0.9).minimize(loss)
```

Now we train the classifier for 100 epochs:

Note that we feed the training data in 1 by 1 (which makes the batch_size=1)

```python
for _ in range(100):
    for i in range(len(x)):
        fit(x=x[i:i+1], y=y[i:i+1])
```

Let's see the result:

```python
print(predict(x=x))
```

#### Result

```text
[[0.04694319 0.96412079]
 [0.93683906 0.05355168]
 [0.92493098 0.06433411]
 [0.07738456 0.93602463]]
```

It's learning!
