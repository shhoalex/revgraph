# Function

A `function_primitive` is another fundamental component of a computational
graph. It binds zero or more value(s) together and transforms it into another `np.tensor`.

Calling the function wraps the arguments (values) into another class that 
derives from `function`. The function itself won't be executed until it's 
applied to `rc.run`.

## Binary Operators

We've actually encountered functions before, the operators `+`, `-`, `*`, `/`
are all `function`s called using python's magic methods.

The library provides functions that are generally useful, such as some of the 
trigonometric functions that are present in the standard library `math`.

## Example

In this example, the graph `add(sin(placeholder(name='x'), constant(2.0))` is 
created.

```python
import revgraph.core as rc
from math import pi

x = rc.placeholder(shape=(), name='x')
f = rc.sin(x) + 2.0

print(f(x=pi/2))
```

### Output
```text
3.0
```
