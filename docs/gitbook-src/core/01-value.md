# Value

A `value` is a fundamental component of a computational graph that extends the functionality of a `rc.tensor`. It has an 
internal property `data` that represents the actual `numpy` array. All 
`value`s act as a wrapper class that form computational graphs with other 
`value` or `function_primitive`.

User can build computational graphs using the 3 subclasses of `value`:
`constant`, `placeholder` and `value`.

## Constant

A `constant` is a value that isn't supposed to change during graph execution.
Manipulation with the `data` property would raise a `ValueError` exception.

```
rc.constant(data)
```

### Arguments

+ `data`: Either a `numpy.ndarray`, `int`, `float` or list of numbers.

### Example
 
```python
import revgraph.core as rc

x = rc.constant([[1, 0],
                 [0, 1]])
y = rc.constant(3)
z = x*y

result = rc.run(z)
print(result)
```

#### Output
```text
[[3. 0.]
 [0. 3.]]
```

## Placeholder

A placeholder is a value whose actual content (the numpy array) is not 
specified until graph is being executed.

```python
rc.placeholder(name, shape)
```

### Arguments

+ `name`: A string as an ID for the node.
+ `shape`: The expected shape of the actual content. (same as numpy shape)
           A dimension can be specified as `None` if the actual size isn't
           yet determined.

### Example

#### Passing in scalar

```python
import revgraph.core as rc

x = rc.placeholder(name='my_placeholder', shape=())  # Use () to specify scalar
f = x+1

print(f(my_placeholdler=3))
# OR
print(rc.run(f, feed_dict={ 'my_placeholder': 3 }))
```

#### Passing variable-sized tensor as argument

```python
import revgraph.core as rc

# would work as long as the value is a 2D tensor with the final 
# dimension's size=2.
x = rc.placeholder(name='x', shape=(None, 2))
f = x+1 

a = f(x=[[0, 1]])  # Valid
b = f(x=[[1, 2], [3, 4]])  # Valid
c = f(x=[1, 2]) # Error
```

## Variable

A `variable` is a value that the `data` property changes during graph execution
(usually by an `optimizer`). The core functionalities of the library revolve 
around it. It has a hidden property called `gradient` that the reverse mode AD
algorithm manipulates.

```python
rc.variable(data)
```

### Arguments

+ `data`: Either a `numpy.ndarray`, `int`, `float` or list of numbers. Whenever
   a tensor is passed in, a copy is created such that all the 
   manipulations will only affect the copy.

### Example

```python
import revgraph.core as rc

x = rc.variable(2.0)
f = x+1

print(f())   # outputs 3.0
x.data *= 2  # direct manipulation of np array (not recommended)
print(f())   # outputs 5.0
```
