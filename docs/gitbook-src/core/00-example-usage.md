# Core Library

The package `revgraph.core` is a toy computational graph library that supports
reverse-mode AD and provides various differentiable operations. It is a 
low-level computational graph builder that wraps up the class `numpy.ndarray`.

Before we dive into all the technical details, here are 2 examples of this 
package's usage.

## Example 1: Defining a computational graph

The following code snippet shows how to define a basic computational graph that
adds 2 variables, then uses the runner to execute and return the result as a 
numpy array.

### Code
```python
import revgraph.core as rc

a = rc.variable(1)
b = rc.variable(2)
c = a+b

ans = rc.run(c)
print(ans)
```

### Output
```text
3.0
```

## Example 2: Function Optimization

A key usage of `revgraph.core` is to find the minimum value of a function by
"tuning" the values of `variable` object.

Optimize 
*f(x,y) = (x<sup>2</sup> + y - 11)<sup>2</sup> + (x + y<sup>2</sup> - 7)
<sup>2</sup>* 
(Himmelblau's Function).

According to Wikipedia, this function has 4 local minima:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Himmelblau_function.svg/1280px-Himmelblau_function.svg.png)

|      x     |      y     |   f(x,y)   | 
| :--------: | :--------: | :--------: |
|   3.0      | 2.0        | 0.0        |
| -2.805118  | 3.131312   | 0.0        |
| -3.779310  | -3.283186  | 0.0        |
| 3.584428   | -1.848126  | 0.0        |

Instead of solving it analytically, the library computes a precise estimation
using gradient descent (in this case).

#### Code

```python
import revgraph.core as rc

x = rc.variable(0.0)
y = rc.variable(0.0)
f = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

step = rc.sgd().minimize(f)  # Define the objective

for _ in range(1000): step()
 
print('x = {:.9f}, y = {:.9f}'.format(x(), y()))
```

#### Output

```text
x = 3.000000000, y = 2.000000000
```

Using a simple construct, the program finds the local minima it in less than 10
lines of code!
