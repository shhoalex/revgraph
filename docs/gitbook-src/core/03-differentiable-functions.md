# Differentiable Functions


**Important: Due to a bug in the implementation, the tensor argument (where the gradients will be propagated)
must not be entered as keyword argument**

For example: 

`rc.sum(my_tensor, axis=1)` and `rc.sum(my_tensor, 1)` are valid, but `rc.sum(xs=my_tensor, axis=1)` is not.

Here is a list of all the implemented differentiable functions:

## Standard Functions

| Name | Arguments | Description |
| -------- | ----- | ----------- |
| `rc.add`, `+` | `a: value`, `b: value` | Adding 2 numbers |
| `rc.sub`, `-` | `a: value`, `b: value` | Subtracting 2 numbers |
| `rc.mul`, `*` | `a: value`, `b: value` | Multiplying 2 numbers |
| `rc.truediv`, `/` | `a: value`, `b: value` | Dividing 2 numbers |
| `rc.floordiv`, `//` | `a: value`, `b: value` | Floor division of 2 numbers |
| `rc.exp` | `x: value` | *e<sup>x</sup>* |
| `rc.log` | `x: value` | *ln(x)* |
| `rc.matmul` | `a: value`, `b: value` | Dot product of 2 matrices |
| `rc.pow`, `**` | `a: value`, `b: value` | *a<sup>b</sup>* |
| `rc.max` | `a: value`, `axis: int = None`, `keepdims: bool = False` | Maximum value in `a` | 
| `rc.min` | `a: value`, `axis: int = None`, `keepdims: bool = False` | Minimum value in `a` |
| `rc.sin` | `x: value` | *sin(x)* |
| `rc.cos` | `x: value` | *cos(x)* |
| `rc.tan` | `x: value` | *tan(x)* |
| `rc.arcsin` | `x: value` | *arcsin(x)* |
| `rc.arcsinh` | `x: value` | *arcsinh(x)* |
| `rc.arccos` | `x: value` | *arccos(x)* |
| `rc.arccosh` | `x: value` | *arccosh(x)* |
| `rc.arctan` | `x: value` | *arctan(x)* |
| `rc.arctanh` | `x: value` | *arctanh(x)* |
| `rc.abs` | `x: value` | *abs(x)* |
| `rc.sqrt` | `x: value` | *sqrt(x)* |
| `rc.square` | `x: value` | *x^2* |

## Functions for tensor manipulation

| Name | Arguments | Description |
| ---- | --------- | ----------- |
| `rc.conv1d` | `a: value`, `b: value`, `padding: 'VALID' or 'SAME'`, `stride = 1` | Convolving 2 3D tensors of shape a: (n, w, in_channels), b: (n', w, out_channels) |
| `rc.conv2d` | `a: value`, `b: value`, `padding: 'VALID' or 'SAME'`, `stride: Tuple[int, int]` | Convolving 2 4D tensors of shape a: (n, w, h, in_channels), b: (n', h', w', out_channels)|
| `rc.flatten` | `xs: value`, `exclude_dim_0: bool = True` | Flatten the tensor |
| `rc.pad` | `xs: value`, `pad_width: Tuple[int, ...]`, `constant_values: float = 0` | Pad the tensor with `constant_values` |
| `rc.reshape` | `xs: value, newshape: Tuple[int, ...]` | Reshape the tensor to `newshape` |
| `rc.sum` | `xs: value`, `axis = -1` | Sum the tensor along axis=`axis` |

## Optimizers

| Name | Arguments |
| ---- | ---------- | 
| `rc.sgd` | `lr=0.001`, `momentum=0.0`, `decay=0.0`, `nesterov=False` |
| `rc.rmsprop` | `lr=0.001`, `rho=0.9`, `epsilon=1e-9`, `decay=0.0` |
| `rc.adagrad` | `lr=0.001`, `epsilon=1e-9`, `decay=0.0` |
| `rc.adadelta` | `lr=1.0`, `rho=0.95`, `epsilon=1e-6`, `decay=0.0` | 
| `rc.adam` | `lr=0.001`, `beta1=0.9`, `beta2=0.999`, `amsgrad=False`, `epsilon=1e-6`, `decay=0.0` |

To use the optimizer, simply call `rc.[Optimizer's Name]([Optimizer's Configs]).minimize(Target Tensor)`.

### Optimizer Example

#### Code

```python
import revgraph.core as rc

x = rc.variable(0.0)
y = x*x + 2*x + 3

step = rc.rmsprop(lr=0.1).minimize(y)

for _ in range(10000):
    # Perform the optimization repeatedly
    rc.run(step)
   
# See the mutated value of x
print(f'y={y()} when x={x()}')
```

#### Result
```text
y=2.0 when x=-1.0
```
