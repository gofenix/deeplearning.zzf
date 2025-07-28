# Layers API Documentation

## Overview

Layers are the fundamental building blocks of neural networks in Deep Learning ZZF. Each layer transforms input data and passes the result to the next layer.

## Base Layer

### `Layer`

Abstract base class for all layers.

#### Methods

##### `call(inputs)`

Forward pass logic for the layer.

**Parameters:**
- `inputs` (Tensor): Input tensor(s).

**Returns:**
- `Tensor`: Output tensor(s).

##### `build(input_shape)`

Creates the layer's weights.

**Parameters:**
- `input_shape` (tuple): Shape of the input.

## Dense Layers

### `Dense`

Fully connected layer.

#### Constructor

```python
Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
      activity_regularizer=None)
```

**Parameters:**
- `units` (int): Number of output units.
- `activation` (str or callable, optional): Activation function to use.
- `use_bias` (bool, default=True): Whether to use bias weights.
- `kernel_initializer` (str or callable, default='glorot_uniform'): Initializer for kernel weights.
- `bias_initializer` (str or callable, default='zeros'): Initializer for bias weights.
- `kernel_regularizer` (str or callable, optional): Regularizer for kernel weights.
- `bias_regularizer` (str or callable, optional): Regularizer for bias weights.
- `activity_regularizer` (str or callable, optional): Regularizer for layer activations.

**Example:**
```python
import deeplearning_zzf as dl

# Basic dense layer
dense1 = dl.layers.Dense(128, activation='relu')

# Dense layer with regularization
dense2 = dl.layers.Dense(
    64, 
    activation='relu',
    kernel_regularizer=dl.regularizers.l2(0.01),
    bias_regularizer=dl.regularizers.l1(0.01)
)

# Output layer for classification
output = dl.layers.Dense(10, activation='softmax')
```

## Convolutional Layers

### `Conv2D`

2D convolution layer.

#### Constructor

```python
Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
       activation=None, use_bias=True, kernel_initializer='glorot_uniform',
       bias_initializer='zeros')
```

**Parameters:**
- `filters` (int): Number of output filters.
- `kernel_size` (int or tuple): Size of the convolution kernel.
- `strides` (int or tuple, default=(1, 1)): Strides of the convolution.
- `padding` (str, default='valid'): Padding strategy ('valid' or 'same').
- `activation` (str or callable, optional): Activation function.
- `use_bias` (bool, default=True): Whether to use bias.
- `kernel_initializer` (str or callable, default='glorot_uniform'): Kernel initializer.
- `bias_initializer` (str or callable, default='zeros'): Bias initializer.

**Example:**
```python
# Basic conv layer
conv1 = dl.layers.Conv2D(32, (3, 3), activation='relu')

# Conv layer with padding and strides
conv2 = dl.layers.Conv2D(
    64, 
    kernel_size=(3, 3), 
    strides=(2, 2), 
    padding='same',
    activation='relu'
)

# Depthwise separable convolution
conv3 = dl.layers.Conv2D(128, (1, 1), activation='relu')
```

### `Conv1D`

1D convolution layer for sequence data.

#### Constructor

```python
Conv1D(filters, kernel_size, strides=1, padding='valid', activation=None)
```

**Example:**
```python
# For sequence processing
conv1d = dl.layers.Conv1D(64, 3, activation='relu', padding='same')
```

## Pooling Layers

### `MaxPooling2D`

Max pooling operation for 2D spatial data.

#### Constructor

```python
MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
```

**Parameters:**
- `pool_size` (int or tuple, default=(2, 2)): Size of the pooling window.
- `strides` (int or tuple, optional): Strides of the pooling operation.
- `padding` (str, default='valid'): Padding strategy.

**Example:**
```python
# Standard max pooling
maxpool = dl.layers.MaxPooling2D((2, 2))

# Custom pooling with strides
maxpool_custom = dl.layers.MaxPooling2D(
    pool_size=(3, 3), 
    strides=(2, 2), 
    padding='same'
)
```

### `AveragePooling2D`

Average pooling operation for 2D spatial data.

**Example:**
```python
avgpool = dl.layers.AveragePooling2D((2, 2))
```

### `GlobalMaxPooling2D`

Global max pooling for 2D spatial data.

**Example:**
```python
global_max = dl.layers.GlobalMaxPooling2D()
```

### `GlobalAveragePooling2D`

Global average pooling for 2D spatial data.

**Example:**
```python
global_avg = dl.layers.GlobalAveragePooling2D()
```

## Recurrent Layers

### `LSTM`

Long Short-Term Memory layer.

#### Constructor

```python
LSTM(units, activation='tanh', recurrent_activation='sigmoid', 
     use_bias=True, dropout=0.0, recurrent_dropout=0.0, 
     return_sequences=False, return_state=False, stateful=False)
```

**Parameters:**
- `units` (int): Number of LSTM units.
- `activation` (str, default='tanh'): Activation function for the cell state.
- `recurrent_activation` (str, default='sigmoid'): Activation for the recurrent step.
- `use_bias` (bool, default=True): Whether to use bias.
- `dropout` (float, default=0.0): Dropout rate for inputs.
- `recurrent_dropout` (float, default=0.0): Dropout rate for recurrent connections.
- `return_sequences` (bool, default=False): Whether to return full sequences.
- `return_state` (bool, default=False): Whether to return the last state.
- `stateful` (bool, default=False): Whether to maintain state between batches.

**Example:**
```python
# Basic LSTM
lstm1 = dl.layers.LSTM(128, return_sequences=True)

# LSTM with dropout
lstm2 = dl.layers.LSTM(
    64, 
    dropout=0.2, 
    recurrent_dropout=0.2,
    return_sequences=False
)

# Bidirectional LSTM
bilstm = dl.layers.Bidirectional(dl.layers.LSTM(128))
```

### `GRU`

Gated Recurrent Unit layer.

#### Constructor

```python
GRU(units, activation='tanh', recurrent_activation='sigmoid',
    dropout=0.0, recurrent_dropout=0.0, return_sequences=False)
```

**Example:**
```python
gru = dl.layers.GRU(128, return_sequences=True, dropout=0.2)
```

### `SimpleRNN`

Simple recurrent neural network layer.

**Example:**
```python
rnn = dl.layers.SimpleRNN(64, return_sequences=False)
```

## Normalization Layers

### `BatchNormalization`

Batch normalization layer.

#### Constructor

```python
BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
```

**Parameters:**
- `axis` (int, default=-1): Axis to normalize.
- `momentum` (float, default=0.99): Momentum for moving average.
- `epsilon` (float, default=1e-3): Small float for numerical stability.
- `center` (bool, default=True): Whether to add offset parameter.
- `scale` (bool, default=True): Whether to add scale parameter.

**Example:**
```python
# Standard batch normalization
bn = dl.layers.BatchNormalization()

# Custom batch normalization
bn_custom = dl.layers.BatchNormalization(
    momentum=0.95, 
    epsilon=1e-5
)
```

### `LayerNormalization`

Layer normalization.

**Example:**
```python
ln = dl.layers.LayerNormalization()
```

## Regularization Layers

### `Dropout`

Dropout layer for regularization.

#### Constructor

```python
Dropout(rate, noise_shape=None, seed=None)
```

**Parameters:**
- `rate` (float): Fraction of inputs to drop (between 0 and 1).
- `noise_shape` (tuple, optional): Shape for generated dropout mask.
- `seed` (int, optional): Random seed.

**Example:**
```python
# Standard dropout
dropout1 = dl.layers.Dropout(0.2)

# Dropout with custom noise shape
dropout2 = dl.layers.Dropout(0.5, noise_shape=(None, 1, 1, 1))
```

## Activation Layers

### `Activation`

Applies an activation function to the input.

#### Constructor

```python
Activation(activation)
```

**Parameters:**
- `activation` (str or callable): Activation function to apply.

**Example:**
```python
# Common activations
relu = dl.layers.Activation('relu')
sigmoid = dl.layers.Activation('sigmoid')
tanh = dl.layers.Activation('tanh')
softmax = dl.layers.Activation('softmax')

# Custom activation
def custom_activation(x):
    return x * dl.backend.sigmoid(x)  # Swish activation

swish = dl.layers.Activation(custom_activation)
```

## Utility Layers

### `Flatten`

Flattens input tensor to 1D.

**Example:**
```python
flatten = dl.layers.Flatten()
```

### `Reshape`

Reshapes input tensor to specified shape.

#### Constructor

```python
Reshape(target_shape)
```

**Parameters:**
- `target_shape` (tuple): Target shape (excluding batch dimension).

**Example:**
```python
# Reshape to 2D
reshape = dl.layers.Reshape((28, 28))

# Reshape to 3D
reshape_3d = dl.layers.Reshape((7, 7, 64))
```

### `Permute`

Permutes the dimensions of input tensor.

#### Constructor

```python
Permute(dims)
```

**Parameters:**
- `dims` (tuple): Permutation pattern.

**Example:**
```python
# Change from (batch, height, width, channels) to (batch, channels, height, width)
permute = dl.layers.Permute((3, 1, 2))
```

### `Lambda`

Wraps arbitrary expressions as a layer.

#### Constructor

```python
Lambda(function, output_shape=None)
```

**Parameters:**
- `function` (callable): Function to be evaluated.
- `output_shape` (tuple, optional): Expected output shape.

**Example:**
```python
# Custom operation
square = dl.layers.Lambda(lambda x: x ** 2)

# With output shape
normalize = dl.layers.Lambda(
    lambda x: x / dl.backend.sqrt(dl.backend.sum(x**2, axis=-1, keepdims=True)),
    output_shape=lambda input_shape: input_shape
)
```

## Complete Example

```python
import deeplearning_zzf as dl

# Build a CNN for image classification
model = dl.Model()

# Feature extraction
model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Dropout(0.25))

model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Dropout(0.25))

# Classification head
model.add_layer(dl.layers.Flatten())
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.Dropout(0.5))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# Compile and summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```