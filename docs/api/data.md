# Data Processing API Documentation

## Overview

The data processing module provides utilities for loading, preprocessing, and augmenting data for machine learning tasks.

## Dataset Classes

### `Dataset`

Base class for all datasets.

#### Methods

##### `__len__()`

Returns the number of samples in the dataset.

##### `__getitem__(index)`

Returns a sample from the dataset.

##### `batch(batch_size)`

Creates batches from the dataset.

**Parameters:**
- `batch_size` (int): Size of each batch.

**Example:**
```python
import deeplearning_zzf as dl

# Custom dataset
class CustomDataset(dl.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Usage
dataset = CustomDataset(x_train, y_train)
batched_dataset = dataset.batch(32)

for batch_x, batch_y in batched_dataset:
    # Process batch
    pass
```

### `TensorDataset`

Dataset wrapping tensors.

#### Constructor

```python
TensorDataset(*tensors)
```

**Parameters:**
- `*tensors`: Variable number of tensors to wrap.

**Example:**
```python
import numpy as np

# Create tensor dataset
x = np.random.random((1000, 28, 28, 1))
y = np.random.randint(0, 10, (1000,))

dataset = dl.data.TensorDataset(x, y)
dataloader = dl.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in dataloader:
    print(f"Batch shape: {batch_x.shape}, {batch_y.shape}")
```

## Data Loaders

### `DataLoader`

Data loader for iterating over datasets.

#### Constructor

```python
DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, 
           drop_last=False, collate_fn=None)
```

**Parameters:**
- `dataset` (Dataset): Dataset to load from.
- `batch_size` (int, default=1): Batch size.
- `shuffle` (bool, default=False): Whether to shuffle data.
- `num_workers` (int, default=0): Number of worker processes.
- `drop_last` (bool, default=False): Whether to drop the last incomplete batch.
- `collate_fn` (callable, optional): Function to collate samples into batches.

**Example:**
```python
# Basic data loader
train_loader = dl.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True,
    num_workers=4
)

# Validation loader (no shuffling)
val_loader = dl.data.DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Data Preprocessing

### `ImageDataGenerator`

Real-time data augmentation for images.

#### Constructor

```python
ImageDataGenerator(rotation_range=0, width_shift_range=0.0, 
                  height_shift_range=0.0, brightness_range=None,
                  shear_range=0.0, zoom_range=0.0, 
                  channel_shift_range=0.0, fill_mode='nearest',
                  horizontal_flip=False, vertical_flip=False,
                  rescale=None, preprocessing_function=None,
                  data_format=None, validation_split=0.0)
```

**Parameters:**
- `rotation_range` (float): Degree range for random rotations.
- `width_shift_range` (float): Range for random horizontal shifts.
- `height_shift_range` (float): Range for random vertical shifts.
- `brightness_range` (tuple, optional): Range for brightness adjustments.
- `shear_range` (float): Shear intensity.
- `zoom_range` (float): Range for random zoom.
- `horizontal_flip` (bool): Whether to randomly flip horizontally.
- `vertical_flip` (bool): Whether to randomly flip vertically.
- `rescale` (float, optional): Rescaling factor.

**Example:**
```python
# Data augmentation for training
train_datagen = dl.data.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255,
    fill_mode='nearest'
)

# No augmentation for validation
val_datagen = dl.data.ImageDataGenerator(rescale=1./255)

# Generate batches
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train model
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)
```

#### Methods

##### `flow(x, y=None, batch_size=32, shuffle=True)`

Generates batches from numpy arrays.

**Parameters:**
- `x` (array): Input data.
- `y` (array, optional): Target data.
- `batch_size` (int): Batch size.
- `shuffle` (bool): Whether to shuffle.

**Example:**
```python
# From numpy arrays
train_generator = train_datagen.flow(
    x_train, y_train,
    batch_size=32,
    shuffle=True
)

for batch_x, batch_y in train_generator:
    # Augmented batch
    pass
```

##### `flow_from_directory(directory, target_size=(256, 256), color_mode='rgb')`

Generates batches from directory structure.

**Parameters:**
- `directory` (str): Path to directory.
- `target_size` (tuple): Target image size.
- `color_mode` (str): Image color mode ('rgb', 'grayscale').

**Example:**
```python
# Directory structure:
# data/
#   train/
#     class1/
#       img1.jpg
#       img2.jpg
#     class2/
#       img3.jpg
#       img4.jpg

generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb'
)
```

### `TextVectorization`

Text preprocessing and vectorization.

#### Constructor

```python
TextVectorization(max_tokens=None, standardize='lower_and_strip_punctuation',
                 split='whitespace', ngrams=None, output_mode='int',
                 output_sequence_length=None, pad_to_max_tokens=False)
```

**Parameters:**
- `max_tokens` (int, optional): Maximum vocabulary size.
- `standardize` (str): Text standardization mode.
- `split` (str): Text splitting mode.
- `output_mode` (str): Output mode ('int', 'multi_hot', 'count', 'tf_idf').
- `output_sequence_length` (int, optional): Maximum sequence length.

**Example:**
```python
# Text vectorization
vectorizer = dl.data.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100,
    output_mode='int'
)

# Adapt to text data
text_data = ["This is a sample text", "Another example sentence"]
vectorizer.adapt(text_data)

# Transform text
sequences = vectorizer(text_data)
print(sequences)  # [[4, 6, 2, 8, 9], [3, 7, 5]]

# Use in model
model = dl.Model()
model.add_layer(vectorizer)
model.add_layer(dl.layers.Embedding(10000, 128))
model.add_layer(dl.layers.LSTM(64))
model.add_layer(dl.layers.Dense(1, activation='sigmoid'))
```

### `Normalization`

Feature normalization layer.

#### Constructor

```python
Normalization(axis=-1, mean=None, variance=None)
```

**Parameters:**
- `axis` (int): Axis to normalize.
- `mean` (array, optional): Precomputed mean.
- `variance` (array, optional): Precomputed variance.

**Example:**
```python
# Feature normalization
normalizer = dl.data.Normalization()

# Adapt to training data
normalizer.adapt(x_train)

# Use in model
model = dl.Model()
model.add_layer(normalizer)
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# Manual normalization
normalized_data = normalizer(x_test)
```

## Data Utilities

### `train_test_split`

Split arrays into train and test sets.

```python
train_test_split(X, y=None, test_size=0.25, train_size=None, 
                random_state=None, shuffle=True, stratify=None)
```

**Parameters:**
- `X` (array): Features.
- `y` (array, optional): Labels.
- `test_size` (float): Proportion of test set.
- `random_state` (int, optional): Random seed.
- `shuffle` (bool): Whether to shuffle before splitting.
- `stratify` (array, optional): Labels for stratified splitting.

**Example:**
```python
import numpy as np

# Generate sample data
X = np.random.random((1000, 20))
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = dl.data.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
```

### `make_batches`

Create batches from data.

```python
make_batches(size, batch_size)
```

**Parameters:**
- `size` (int): Total size of data.
- `batch_size` (int): Size of each batch.

**Returns:**
- `list`: List of (start, end) tuples for each batch.

**Example:**
```python
# Create batch indices
batch_indices = dl.data.make_batches(1000, 32)

for start, end in batch_indices:
    batch_x = x_data[start:end]
    batch_y = y_data[start:end]
    # Process batch
```

### `shuffle`

Shuffle arrays in unison.

```python
shuffle(*arrays, random_state=None)
```

**Example:**
```python
# Shuffle data
X_shuffled, y_shuffled = dl.data.shuffle(X, y, random_state=42)
```

## Built-in Datasets

### `load_mnist`

Load MNIST handwritten digits dataset.

```python
load_mnist(path='mnist.npz')
```

**Returns:**
- `tuple`: (x_train, y_train), (x_test, y_test)

**Example:**
```python
# Load MNIST
(x_train, y_train), (x_test, y_test) = dl.data.load_mnist()

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
print(f"Train labels: {y_train.shape}, Test labels: {y_test.shape}")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert to categorical
y_train = dl.utils.to_categorical(y_train, 10)
y_test = dl.utils.to_categorical(y_test, 10)
```

### `load_cifar10`

Load CIFAR-10 dataset.

**Example:**
```python
(x_train, y_train), (x_test, y_test) = dl.data.load_cifar10()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```

### `load_imdb`

Load IMDB movie reviews dataset.

```python
load_imdb(num_words=None, skip_top=0, maxlen=None, seed=113)
```

**Parameters:**
- `num_words` (int, optional): Top words to keep.
- `skip_top` (int): Skip top N frequent words.
- `maxlen` (int, optional): Maximum sequence length.

**Example:**
```python
# Load IMDB reviews
(x_train, y_train), (x_test, y_test) = dl.data.load_imdb(
    num_words=10000,
    maxlen=500
)

# Pad sequences
x_train = dl.data.pad_sequences(x_train, maxlen=500)
x_test = dl.data.pad_sequences(x_test, maxlen=500)
```

## Sequence Processing

### `pad_sequences`

Pad sequences to the same length.

```python
pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre',
             truncating='pre', value=0.0)
```

**Parameters:**
- `sequences` (list): List of sequences.
- `maxlen` (int, optional): Maximum length.
- `padding` (str): Padding mode ('pre' or 'post').
- `truncating` (str): Truncating mode ('pre' or 'post').
- `value` (float): Padding value.

**Example:**
```python
# Pad text sequences
sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
padded = dl.data.pad_sequences(sequences, maxlen=5, padding='post')
print(padded)
# [[1 2 3 0 0]
#  [4 5 0 0 0]
#  [6 7 8 9 10]]
```

### `TimeseriesGenerator`

Generate batches of temporal data.

#### Constructor

```python
TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1,
                   start_index=0, end_index=None, shuffle=False,
                   reverse=False, batch_size=128)
```

**Parameters:**
- `data` (array): Input time series data.
- `targets` (array): Target data.
- `length` (int): Length of input sequences.
- `sampling_rate` (int): Sampling rate within sequences.
- `stride` (int): Stride between successive sequences.
- `shuffle` (bool): Whether to shuffle sequences.
- `batch_size` (int): Batch size.

**Example:**
```python
import numpy as np

# Generate time series data
time_steps = 1000
data = np.sin(np.linspace(0, 100, time_steps))
targets = np.cos(np.linspace(0, 100, time_steps))

# Create generator
generator = dl.data.TimeseriesGenerator(
    data, targets,
    length=50,  # Use 50 time steps to predict next value
    batch_size=32
)

# Use with model
model = dl.Model()
model.add_layer(dl.layers.LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add_layer(dl.layers.LSTM(50))
model.add_layer(dl.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(generator, epochs=50)
```

## Data Pipeline Example

### Complete Image Classification Pipeline

```python
import deeplearning_zzf as dl
import numpy as np

# 1. Data Loading and Preprocessing
def create_data_pipeline(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    # Training data with augmentation
    train_datagen = dl.data.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data without augmentation
    val_datagen = dl.data.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

# 2. Text Processing Pipeline
def create_text_pipeline(texts, labels, max_features=10000, max_len=100):
    # Vectorize text
    vectorizer = dl.data.TextVectorization(
        max_tokens=max_features,
        output_sequence_length=max_len,
        output_mode='int'
    )
    
    vectorizer.adapt(texts)
    
    # Convert to sequences
    sequences = vectorizer(texts)
    
    # Split data
    x_train, x_test, y_train, y_test = dl.data.train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    return x_train, x_test, y_train, y_test, vectorizer

# 3. Time Series Pipeline
def create_timeseries_pipeline(data, sequence_length=60, test_size=0.2):
    # Normalize data
    normalizer = dl.data.Normalization()
    normalizer.adapt(data)
    normalized_data = normalizer(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(normalized_data)):
        X.append(normalized_data[i-sequence_length:i])
        y.append(normalized_data[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, normalizer

# Usage examples
if __name__ == "__main__":
    # Image classification
    train_gen, val_gen = create_data_pipeline('data/train', 'data/val')
    
    # Text classification
    texts = ["positive review", "negative review"]
    labels = [1, 0]
    x_train, x_test, y_train, y_test, vectorizer = create_text_pipeline(texts, labels)
    
    # Time series prediction
    time_data = np.random.random((1000, 1))
    X_train, X_test, y_train, y_test, normalizer = create_timeseries_pipeline(time_data)
```