# Core API Documentation

## Overview

The core API provides the fundamental building blocks for the Deep Learning ZZF framework.

## Classes

### `Model`

The main model class for building and training neural networks.

#### Constructor

```python
Model(name=None, dynamic=False)
```

**Parameters:**
- `name` (str, optional): Name for the model. If None, a unique name will be generated.
- `dynamic` (bool, default=False): Whether to enable dynamic graph execution.

**Example:**
```python
import deeplearning_zzf as dl

# Create a basic model
model = dl.Model(name="my_classifier")

# Create a dynamic model
dynamic_model = dl.Model(dynamic=True)
```

#### Methods

##### `add_layer(layer)`

Adds a layer to the model.

**Parameters:**
- `layer` (Layer): A layer instance to add to the model.

**Returns:**
- `Model`: Returns self for method chaining.

**Example:**
```python
model = dl.Model()
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.Dropout(0.2))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
```

##### `compile(optimizer, loss, metrics=None)`

Configures the model for training.

**Parameters:**
- `optimizer` (str or Optimizer): Optimization algorithm to use.
- `loss` (str or Loss): Loss function to minimize.
- `metrics` (list, optional): List of metrics to monitor during training.

**Example:**
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)
```

##### `fit(x, y, batch_size=32, epochs=1, validation_data=None, callbacks=None)`

Trains the model on the given data.

**Parameters:**
- `x` (array-like): Input data.
- `y` (array-like): Target data.
- `batch_size` (int, default=32): Number of samples per gradient update.
- `epochs` (int, default=1): Number of epochs to train the model.
- `validation_data` (tuple, optional): Data on which to evaluate the loss and metrics.
- `callbacks` (list, optional): List of callbacks to apply during training.

**Returns:**
- `History`: Training history object.

**Example:**
```python
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[dl.callbacks.EarlyStopping(patience=10)]
)
```

##### `predict(x, batch_size=None)`

Generates predictions for the input samples.

**Parameters:**
- `x` (array-like): Input data.
- `batch_size` (int, optional): Size of batches for prediction.

**Returns:**
- `array`: Predictions.

**Example:**
```python
predictions = model.predict(x_test, batch_size=128)
```

##### `evaluate(x, y, batch_size=None)`

Evaluates the model on test data.

**Parameters:**
- `x` (array-like): Input data.
- `y` (array-like): Target data.
- `batch_size` (int, optional): Size of batches for evaluation.

**Returns:**
- `dict`: Dictionary containing loss and metrics values.

**Example:**
```python
results = model.evaluate(x_test, y_test, batch_size=128)
print(f"Test accuracy: {results['accuracy']:.4f}")
```

##### `save(filepath)`

Saves the model to a file.

**Parameters:**
- `filepath` (str): Path where to save the model.

**Example:**
```python
model.save('my_model.h5')
```

##### `load(filepath)` (class method)

Loads a model from a file.

**Parameters:**
- `filepath` (str): Path to the saved model.

**Returns:**
- `Model`: Loaded model instance.

**Example:**
```python
loaded_model = dl.Model.load('my_model.h5')
```

## Functions

### `load_model(filepath)`

Utility function to load a saved model.

**Parameters:**
- `filepath` (str): Path to the saved model.

**Returns:**
- `Model`: Loaded model instance.

**Example:**
```python
model = dl.load_model('trained_model.h5')
```

### `set_random_seed(seed)`

Sets the random seed for reproducible results.

**Parameters:**
- `seed` (int): Random seed value.

**Example:**
```python
dl.set_random_seed(42)
```

## Constants

### `__version__`

String containing the version number of the framework.

```python
print(f"Deep Learning ZZF version: {dl.__version__}")
```