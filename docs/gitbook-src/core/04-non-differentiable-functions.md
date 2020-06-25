# Non-Differentiable Functions

`NoGradFunction` is a class that generates functions that won't propagate its gradient to its child nodes. Most of the comparison functions are non-differentiable.

Here's a list of non-differentiable functions.

| Name | Arguments | Description |
| ---- | --------- | ----------- |
| `rc.len` | `xs: tensor` | Dynamically evaluates the length (dim0) of a tensor at runtime |
| `>` | `a: tensor`, `b: tensor` | a>b |
| `<` | `a: tensor`, `b: tensor` | a<b |
| `>=` | `a: tensor`, `b: tensor` | a>=b |
| `<=` | `a: tensor`, `b: tensor` | a<=b |
| `==` | `a: tensor`, `b: tensor` | a==b |
| `!=` | `a: tensor`, `b: tensor` | a!=b |
| `rc.all` | `a: tensor` | Same as `np.all` |
| `rc.any` | `a: tensor` | Same as `np.any` |
| `rc.argmax` | `a: tensor`, `axis = -1` | Same as `np.argmax` |
| `rc.argmin` | `a: tensor`, `axis = -1` | Same as `np.argmin` |

# Custom Differentiable Function

The `@rc.no_grad` decorator allows the creation of custom 
non-differentiable functions.

### `no_grad` example

#### Code

```python
import revgraph.core as rc

@rc.no_grad
def f(x):
    print('Running')
    return x*2

# Defining the graph
a = rc.constant(3)
b = rc.constant(4)
c = a + f(b)

# Run the node c
print('Before Evaluation')
d = rc.run(c)  # Dynamically evaluates the function f
print('After Evaluation')
print('result is', d)
```

#### Result
```text
Before Evaluation
Running
After Evaluation
result is 11.0
```


