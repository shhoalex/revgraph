# Deep Learning Library

The package `revgraph.dl` is a toy deep learning library that provides a
keras-like interface for defining deep learning models. It is built 
entirely on top of `revgraph.core` such that the interfaces are merely
graph builders and wrappers of `rc.tensor`.

### Quick Example: XOR MLP

Here is a simple MLP (almost identical with the one in the [previous post](../core/05-a-simple-multilayer-perceptron.md)) but defined using 
`revgraph.dl` instead of `revgraph.core`.

```python
import revgraph.dl as dl

# Training Data
x = [[0,0],
     [0,1], 
     [1,0], 
     [1,1]]

y = [[0,1], 
     [1,0], 
     [1,0], 
     [0,1]]

architecture = dl.sequential(
    dl.inputs(2),
    dl.dense(8, activation='tanh'),
    dl.dense(2, activation='sigmoid')
)

model = dl.Model(
    model=architecture,
    loss=dl.mean_squared_error(),
    optimizer=dl.sgd(lr=0.1, momentum=0.9)
)

model.compile()  # Compile specifications to a graph
model.fit(x, y, epochs=100, batch_size=1)
print(model.predict(x))
```

As you can see, the library abstracts away the creation of low level 
tensors and focuses on the architecture of the neural network itself.
