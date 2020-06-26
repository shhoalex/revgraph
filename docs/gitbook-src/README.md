# revgraph

A toy deep learning library built using `numpy` as its only dependency (and 
`dill` for pickling the computational graph).


### Motivation

I hacked together this project just to deepen my own understanding of how deep
learning frameworks work under the hood. **It is probably the most inefficient 
implementation you could have ever found (lol), so please don't use it for 
anything real.** It's simply a hackable toy computational graph / deep 
learning library that tries to replicate some of tensorflow and keras' 
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

### Full Documentation

Read the full documentation [here]().
