# Putting it all together - MNIST CNN

To summarize, we are going to build a convolutional neural network and train 
it on the MNIST dataset for handwritten digit classification.

## Part 1 - Training the Model

### Import DL

```python
import revgraph.dl as dl
```

### Load the training data

The library also ships with the mnist dataset for users to mess around
with it.

```python
import revgraph.datasets.mnist as mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


### Defining the model

Just a random CNN configuration...

```python
layers = dl.sequential(
    dl.inputs(784),
    dl.reshape((28, 28, 1)),
    dl.conv2d(
        filters=12,
        kernel_size=(3,3),
        padding='same',
        activation='relu',
        use_bias=True,
        kernel_regularizer=dl.l1(),
        bias_regularizer=dl.l1(),
        bias_initializer='zeros'
    ),
    dl.flatten(),
    dl.dense(
        units=128,
        activation='relu',
        use_bias=True,
        kernel_regularizer=dl.l1(),
        bias_regularizer=dl.l1()
    ),
    dl.dense(
        units=10,
        activation='softmax',
        use_bias=True,
        kernel_regularizer=dl.l1(),
        bias_regularizer=dl.l1()
    )
)
```

### Training Configuration

Again, just some standard stuffs...

```python
model = dl.Model(
    model=layers,
    loss=dl.categorical_cross_entropy(),
    optimizer=dl.adam(lr=0.01, decay=0.01),
    metrics={
        'Accuracy': dl.categorical_accuracy()
    }
)
```

### Compiling the model

+ Note that the validation set used is the remaining 15% of `(x_train, y_train)`,
  while the test set is stored in a completely different variable `(x_test, y_test)`.


```python
model.compile()

model.fit(
    x=x_train,
    y=y_train,
    epochs=3,
    batch_size=500,
    train_test_validation=(0.85, 0, 0.15),
    callbacks=[dl.callbacks.progress(),
               dl.callbacks.validation(),
               dl.callbacks.test(x_test=x_test, y_test=y_test),
               dl.callbacks.save('cnn.pkl')]
)
```

### Result

+ After 3 epochs (crazy slow I know :( ), the model achieved 98.13% accuracy
  on test set, not too bad!

```text
Training Model tensor(addr=0x10dfdf650) with 51000 records
  + Epoch 1/3
    [##########] (100.0%)
    Time Elapsed: 101.355s
    Validation Accuracy: 0.978
  + Epoch 2/3
    [##########] (100.0%)
    Time Elapsed: 97.104s
    Validation Accuracy: 0.9828888888888889
  + Epoch 3/3
    [##########] (100.0%)
    Time Elapsed: 96.655s
    Validation Accuracy: 0.9836666666666667
Training Completed.
Total Time Elapsed: 337.059s
Test Accuracy: 0.9813
Saving session <revgraph.dl.model.model.Model object at 0x10de82390> to 'cnn.pkl'
```

## Part 2 - Visualizing the results

In a new file, we load the saved model (and also the test dataset for testing).

```python
import revgraph.dl as dl
from revgraph.datasets.mnist import load_data

_, (images, _) = load_data()
```

### Visualization function

We define a small helper function for "displaying" the images 
(it's just me being too lazy to add `matplotlib` as dependency :P).


```python
def print_img(img):
    img = img.reshape(28, 28)
    print('--' * 30)
    for row in img:
        print('| ', end='')
        for cell in row:
            print('##' if cell>0.5 else '  ', end='')
        print(' |')
    print('--' * 30)
```

### Prediction Function

We use numpy's argmax function to see which digit is the 
model predicting (since the higher the value the more 
probable that the image belongs to that class).

```python
import numpy as np

def show_and_predict(img):
    dist = model.predict_one(img)
    digit = np.argmax(dist)
    print_img(img)
    print(f"It's a '{digit}'!")
```

### Sampling

Now we sample 3 image as our target and predict them.

```python
indices = np.random.choice([i for i in range(10000)], 5)

for index in indices:
    show_and_predict(images[index])
```

```text
------------------------------------------------------------
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                           ####           |
|                                       ########           |
|                                     ########             |
|                                     ########             |
|                                   ########               |
|                                 ########                 |
|                                 ######                   |
|                                 ####                     |
|                               ####                       |
|                             ####                         |
|                           ####                           |
|                       ########                           |
|                     ########                             |
|                   ########                               |
|                 ########                                 |
|                 ######                                   |
|               ########                                   |
|               ######                                     |
|             ########                                     |
|             ######                                       |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
------------------------------------------------------------
It's a '1'!
------------------------------------------------------------
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|               ####################                       |
|           ############################                   |
|           ##############################                 |
|                                 ##########               |
|                                     ######               |
|                                     ######               |
|                                   ########               |
|                                   ########               |
|                                 ########                 |
|                               ########                   |
|                               ########                   |
|                             ########                     |
|                           ########                       |
|                         ##########                       |
|                       ##########                         |
|                     ##########                           |
|                     ########                             |
|                     ######                               |
|                   ########                               |
|                     ####                                 |
|                                                          |
------------------------------------------------------------
It's a '7'!
------------------------------------------------------------
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
|                               ############               |
|                       ##########################         |
|                   ##################################     |
|                 ##############                           |
|               ##########                                 |
|               ########                                   |
|                 ##########                               |
|                   ############                           |
|                       ##########                         |
|                               ####                       |
|                               ####                       |
|                                 ####                     |
|                                 ####                     |
|                             ########                     |
|                         ##########                       |
|             ##################                           |
|               ############                               |
|                                                          |
|                                                          |
|                                                          |
|                                                          |
------------------------------------------------------------
It's a '5'!
```

Yeahh it works :)
