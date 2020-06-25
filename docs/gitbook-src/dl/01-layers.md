# Layers

`layer` is the heart of the entire library. It specifies how to construct 
low level tensors based on the parameters (or specifications) given.

## Dense Layer

The `dense` layer represents a fully connected network. It basically
creates the tensor with structure similar to 
`activation(previous_layer.dot(weights) + bias)`.

**Parameters**

+ `units: int`: Number of units in this layer.
+ `use_bias: bool = True`: Create bias units or not
+ `activation: ActivationFunction = 'linear'`: Activation Function used
+ Initializers
    * `kernel_initializer: Initializer = 'glorot_normal'`
    * `bias_initializer: Initializer = 'glorot_normal`
+ Regularizers
    * `kernel_regularizer: Regularizer = None`: Method used to regularize weights.
    * `bias_regularizer: Regularizer = None`: Method used to regularize bias.
    * `activity_regularizer: Regularizer = None`: Method used to regularize output (after applying activation function).
 
## Input Layer

The `inputs` layer represents the first layer of any architecture. 
Hence, every definition should start with this layer (unlike keras
where you can use the `input_shape` argument).

Upon the creation of this layer, the placeholder with name=`x` is 
created and its shape equals to `(None,) + shape`.

**Parameters**

+ `shape: Union[Tuple[int, ...], int]`: Specifies the input shape, adding `None` to the first dimension is unnecessary.

## Transform Layer

The `transform` layer accepts a function with type `tensor -> tensor`
and applies it to the output of the previous layer.

**Parameters**

+ `f: TensorFunction`: A function (can be a `revgraph.core` primitive) or simply a function that produces another graph from the
previous layer.

**For example:**

```python
dl.sequential(
    ...,
    dl.dense(10),
    dl.transform(lambda x: x*2),
    ...
)
```

The output of this layer will be the transformed (doubled in this case) version of the previous layer.

## Reshape Layer

The `reshape` layer reshapes the output to `new_shape`.

**Parameters**

+ `new_shape: Union[Tuple[int, ...], int]`: The new shape of the previous layer.

It is basically the same as applying `dl.transform(lambda xs: rc.reshape(xs, newshape))`.

## Flatten Layer

The `flatten` layer flattens the previous layer.

**Parameters**

None

## Conv2D Layer

The `conv2d` layer applies 2D convolution on the previous layer and 
the `filters`.

The output of previous layer must be a 4D tensor of shape (batch_size, h, w, in_channel).

**Parameters**

+ `filters: int`: Number of filters.
+ `kernel_size: Tuple[int, int]`: Size of each filter.
+ `stride: Tuple[int, int] = (1,1)`: The step of each operation.
+ `padding: str = 'valid' or 'same'`: Whether the output keeps the same height and width.
+ `activation: ActivationFunction = 'linear'`: Activation Function used.
+ `use_bias: bool = False`: Create bias units or not.
+ Initializers
    * `kernel_initializer: Initializer = 'glorot_normal'`: Kernel means the filters
    * `bias_initializer: Initializer = 'glorot_normal'`
+ Regularizers
    * `kernel_regularizer: Regularizer = None`: Method used to regularize the filters.
    * `bias_regularizer: Regularizer = None`: Method used to regularize bias.
    * `activity_regularizer: Regularizer = None`: Method used to regularize output (after applying activation function).

**Example**

```python
dl.sequential(
    ...,
    dl.conv2d(
        filters=6,  # Initialize 6 filters
        kernel_size=(3,3),  # 3x3 filter
        padding='same',  # Output has the same height and width
    )
    ...,
)
```

The output is a 4D tensor of shape `(batch_size, new_height, new_width, filters)`.

## Sequential Layer

The `sequential` layer composes different layers into 1 big layer.

**Parameter**

+ `*layers: GraphBuilder`: Different layers in sequence.

**Alternative**

Instead of using sequential layer, previous layer can be prepend to 
current layer by calling the current layer with the previous one
as its argument. For example:

```python
import revgraph.dl as dl

l0 = dl.inputs(2)()
l1 = dl.dense(10)(l0)
l2 = dl.transform(lambda x: (x>0) * x)(l1)
l3 = lambda: dl.dense(2)(l2)

# l4 and l3 are basically equivalent.

l4 = dl.sequential(
    dl.inputs(2),
    dl.dense(10),
    dl.transform(lambda x: (x>0) * x),
    dl.dense(2)
)
```

This is similar to kera's functional API. However, having multiple 
layers as input is current unsupported since the Add (or Merge)
layer hasn't been implemented yet.
