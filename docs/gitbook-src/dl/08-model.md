# Model

A `Model` is a class that binds everything (the layers, loss function and optimizer) together
and provides a unified interface for training it.

### Initializing a Model

To create a model, simply pass the layers, loss function, metrics and the optimizer as parameters.

```python
import revgraph.dl as dl
import revgraph.core as rc


layers = dl.sequential(
    dl.inputs(784),
    dl.dense(9, use_bias=True)
)

my_model = dl.Model(
    model=layers,
    loss=dl.mean_squared_error(),
    optimizer=dl.sgd(lr=0.01),
    metrics={
        'Accuracy': dl.categorical_accuracy(),
        
        # Manually defined metric: mean absolute error
        'My MAE': lambda y_true, y_pred: 
                    rc.sum(rc.abs(y_true - y_pred)) / rc.len(y_true)
    }
)
``` 

Note that for metrics you have to provide a dictionary with `(name_of_metric, metric)` pair.


### Compiling the model

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
 
+ `verbose: bool = True`: whether to print out [callbacks](./08-callbacks.md).
+ `callbacks: List[base_callback]`: list of [callbacks](./08-callbacks.md).

### Loading a previously trained model

Using the `save` [callback](./08-callbacks.md), models can be saved to a file.

To load it, simply use the static method `dl.Model.load_from(path)`.
Note that `.compile` can't be called if the model is loaded from a file.

#### Example

```python
import revgraph.dl as dl

my_model = dl.Model.load_from(PATH_TO_MY_MODEL)

my_model.fit(
    ...
)

my_model.predict(
    ...
)
```

## Prediction

#### `.predict`

The method `.predict` can be used for prediction.

```python
model = ...  # Suppose this is the xor classifier

print(model.predict([[0, 1], [1, 1]]))
```

#### Output

```text
[[0.9787839  0.02086227]
 [0.02456557 0.97530021]]
```

#### `.predict_one`

A shorthand `.predict_one` is also provided for predicting 1 record.

```python
model = ...

print(model.predict_one([0, 1]))
```

#### Output

```text
[0.97593324 0.02336319]
```
