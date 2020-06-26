# revgraph

A toy deep learning library built using `numpy` as its only dependency (and 
`dill` for pickling the computational graph).

![](https://travis-ci.org/shhoalex/revgraph.svg?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


### Motivation

I hacked together this project just to deepen my own understanding of how deep
learning frameworks work under the hood. **It is probably the most inefficient 
implementation you could have ever found (lol), so please don't use it for 
anything real.** It's simply a hackable toy computational graph / deep 
learning library that tries to replicate some of tensorflow and kera's 
interfaces.


### Key Features

* Gradient Descent Based Optimization Methods
* Regressions
* Deep Learning


### Packages

The project is divided into 2 main parts:

1. `revgraph.core`: A simple computational graph library that supports reverse 
   mode AD.
2. `revgraph.dl`: A high-level deep learning library built entirely on top of 
   `revgraph.core`.


### Usage

#### Finding a function's local minima using gradient descent:

```python
import revgraph.core as rc

x = rc.variable(0.0)
y = 4*x*x - 32*x - 10
optimization = rc.sgd(lr=0.1).minimize(y)

for _ in range(50): 
    rc.run(optimization)

required_y = rc.run(y)
required_x = rc.run(x)

print(f'minimum y = {required_y} when x = {required_x}')
```

##### Output

```text
minimum y = -74.0 when x = 4.0
```

#### Deep Learning

```python
import revgraph.dl as dl

layers = dl.sequential(
    dl.inputs(784),
    dl.dense(
        units=100,
        activation='tanh',
        kernel_initializer='glorot_normal',
        kernel_regularizer=dl.l1(0.01)
    ),
    dl.dense(10, activation='softmax')
)

model = dl.Model(
    model=layers,
    loss=dl.categorical_cross_entropy(),
    optimizer=dl.adam(lr=0.005)
)

...
```


### Credits

The structure/implementation of this library draws heavily on the following 
projects, please don't forget to check them out!

+ `tensorflow`: The computational graph's API is modelled after this.
+ `keras`: The high level API is modelled after kera's API and the 
  implementation of the optimizers is more or less copied from keras.
+ [`autograd`](https://github.com/HIPS/autograd): The computational graph 
  library's `unbroadcast` function is basically how the tensor's able to 
  broadcast its gradient to different shapes.
+ [`conv2d`](https://github.com/renmengye/np-conv2d/tree/master): the implementation
  of the operation Conv2D is based on this implementation.

### Documentation

Read the full documentation [here](https://shhoalex.github.io/revgraph/).
