# revgraph

A minimal deep learning library built using `numpy` as its only dependency.

The project is divided into 2 main parts:

1. `revgraph.core`: A simple computational graph library.
2. `revgraph.dl`: A high-level keras-like deep learning library built entirely
   on top of `revgraph.core`.

Both packages can be used separately.

## Key Features

* Computational Graph
* Automatic Differentiation
* Linear / Logistic Regression
* Deep Learning

## Usage

Before we go into all the technical details, you might want to first get a 
taste of what this library can do.

### `core`

A key usage of `revgraph.core` is to find the minimum value of a function by
"tuning" the values of `variable` objects.

### Example: Function Optimization

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Himmelblau_function.svg/1280px-Himmelblau_function.svg.png)

Optimize $$ f(x,y) = (x^2 + y - 11) + (x + y^2 - 7)^2 $$ (Himmelblau's Function).

According to Wikipedia, this function has 4 local minima:

$$ f(3.0, 2.0) = 0.0 $$

$$ f(-2.805118, 3.131312) = 0.0 $$

$$ f(-3.779310, -3.283186) = 0.0 $$

$$ f(3.584428, -1.848126) = 0.0 $$

Instead of solving it analytically, the library computes a precise estimation
using gradient descent (in this case).

#### Code

```python
import revgraph.core as rc

# Defining computational graph
x = rc.variable(0)
y = rc.variable(0)
f = (x*x + y - 11) ** 2 + (x + y*y - 7) ** 2

# Optimization Algorithm
step = rc.sgd().minimize(f)
train = rc.simple_loop(10000, step)

# Perform the optimization
rc.run(train)

# For the sake of clarity, round the answer to 6 s.f.
ans_x = rc.run(x).round(6)
ans_y = rc.run(y).round(6)
ans_f = rc.run(f).round(6)

print('f({}, {}) = {}'.format(ans_x, ans_y, ans_f))
```

#### Output

```text
f(3.0, 2.0) = 0.0
```

Using the simplest model, the program solves it in 20 lines of code with the
actual difference $$ < 10^{-14} $$!
