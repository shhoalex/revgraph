# Callbacks

A `callback` extends `.fit` method's functionality and can be used by 
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

### `dl.callbacks.progress()`

Progress shows the current training progress and the time elapsed
after training/each epoch. For example:

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
