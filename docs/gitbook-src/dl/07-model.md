# Model

A `Model` is a class that binds everything (the layers, loss function and optimizer) together
and provides a unified interface for training it.

### Initializing a Model

To create a model, simply pass the layers, loss function and the optimizer as parameters.

```python
import revgraph.dl as dl

layers = dl.sequential(
    dl.inputs(784),
    dl.dense(9, use_bias=True)
)

my_model = dl.Model(
    model=layers,
    loss=dl.mean_squared_error(),
    optimizer=dl.sgd(lr=0.01)
)
``` 

At this point, the model is merely specifications of how to build the computational graph,
no tensor has been created at this point. To actually build the graph, 
call the `.compile` method.

```python
my_model.compile()
```

### Fitting the model

To actually start training the model, use the `.fit` method.

**Parameters**

+ `x: np.ndarray`: the training data (inputs).
+ `y: np.ndarray`: the training data (outputs).
+ `epochs: int = 1`: number of rounds to train.
+ `batch_size: Optional[int] = None`: batch size.
+ `shuffle: bool = True`: whether to shuffle `x` and `y` before every epoch.
+ `train_test_validation = (1.0, 0.0, 0.0)`: dividing `(x,y)` into `(x_train, y_train), (x_test, y_test), (x_validation, y_validation)` (`sum(train_test_validation)` must be `1.0`).
 
+ `verbose: bool = True`: whether to print out [callbacks]().
+ `callbacks: List[base_callback]`: list of [callbacks]().
