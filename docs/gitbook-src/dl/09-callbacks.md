# Callbacks

A `callback` extends `.fit`'s functionality and can be used by 
adding them to the `callbacks` parameter of `.fit()`. For example,

```python
my_model.fit(
    callbacks=[
        dl.callbacks.progress(),
        dl.callbacks.another_callback()
    ],
    ...
)
```

## `dl.callbacks.progress`

Progress shows the current training progress and the time elapsed
after training/each epoch. 

**Parameters**

None

**For example:**

Suppose the fit configurations are as below:

```python
model.fit(
    x=x,
    y=y,
    epochs=5,
    batch_size=1,
    callbacks=[
        dl.callbacks.progress()
    ]
)
```
#### Output

```text
Training Model tensor(addr=0x118011810) with 4 records
  + Epoch 1/5
    [##########] (100.0%)
    Time Elapsed: 0.002s
  + Epoch 2/5
    [##########] (100.0%)
    Time Elapsed: 0.002s
  + Epoch 3/5
    [##########] (100.0%)
    Time Elapsed: 0.002s
  + Epoch 4/5
    [##########] (100.0%)
    Time Elapsed: 0.002s
  + Epoch 5/5
    [##########] (100.0%)
    Time Elapsed: 0.003s
Training Completed.
Total Time Elapsed: 0.012s
```

## `dl.callbacks.validation`

**Parameters**

+ `after_every: int = 1`: Perform validation after every n epoch (1 means after every epoch).
+ `x_validation: np.ndarray = None`: specifies the inputs of the training data. If it's `None`, the portion divided in `train_test_validation` will be used.
+ `y_validation: np.ndarray = None`: specifies the outputs of the training data. If it's `None`, the portion divided in `train_test_validation` will be used.

#### Output

After`after_every` epoch, the `metrics` will be evaluated on the validation set. 

For example, when used with `progress` and `metrics` is `{ 'Accuracy': dl.categorical_accuracy() }`:

```text
Training Model tensor(addr=0x10dabd850) with 4 records
  + Epoch 1/3
    [##########] (100.0%)
    Time Elapsed: 0.002s
    Validation Accuracy: 0.5
  + Epoch 2/3
    [##########] (100.0%)
    Time Elapsed: 0.002s
    Validation Accuracy: 0.25
  + Epoch 3/3
    [##########] (100.0%)
    Time Elapsed: 0.002s
    Validation Accuracy: 0.75
Training Completed.
Total Time Elapsed: 0.007s
[[0.4170601  0.45031412]
 [0.54590443 0.52044246]
 [0.40001153 0.56573361]
 [0.51021093 0.61924151]]
```

## `dl.callbacks.test`

Similar to `validation`, but instead of testing the validation set
at the end of certain epochs, it tests the test set at the end of
training.

**Parameters**

+ `x_test: np.ndarray = None`: specifies the inputs of the training data. If it's `None`, the portion divided in `train_test_validation` will be used.
+ `y_test: np.ndarray = None`: specifies the outputs of the training data. If it's `None`, the portion divided in `train_test_validation` will be used.

Suppose we train the xor classifier for 300 epochs and use the test
callback on the entire x and y.

#### Output

```text
...
Training Completed.
Total Time Elapsed: 0.645s
Test Accuracy: 1.0
[[0.01598295 0.98440468]
 [0.9871009  0.0150855 ]
 [0.98141776 0.01882638]
 [0.01973683 0.97951813]]
```

## `dl.callbacks.save`

This callback saves the entire trained model after certain epochs and after training. It saves the entire tensor using `dill` as a
.pkl file (extremely inefficient I know).

**Parameters**

+ `path: str`: string representing the path of the file.
+ `save_after_every: int = None`: Save after every n epochs. If `None`, the default behaviour is to save the model after training.
+ `save_after_all: bool = True`: Save after training.

Suppose the callback `dl.callbacks.save('my_model.pkl')` is added.
The file `my_model.pkl` will be created once the training ends.
