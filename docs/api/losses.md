# Loss Functions API Documentation

## Overview

Loss functions measure the difference between predicted and actual values, providing the error signal that drives the learning process in neural networks.

## Base Loss

### `Loss`

Abstract base class for all loss functions.

#### Methods

##### `call(y_true, y_pred)`

Computes the loss value.

**Parameters:**
- `y_true` (Tensor): Ground truth values.
- `y_pred` (Tensor): Predicted values.

**Returns:**
- `Tensor`: Loss value.

## Regression Losses

### `MeanSquaredError`

Mean squared error loss for regression tasks.

#### Constructor

```python
MeanSquaredError(reduction='auto', name='mean_squared_error')
```

**Parameters:**
- `reduction` (str, default='auto'): Type of reduction to apply ('auto', 'none', 'sum', 'sum_over_batch_size').
- `name` (str, default='mean_squared_error'): Name for the loss.

**Mathematical Formula:**
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**Example:**
```python
import deeplearning_zzf as dl
import numpy as np

# Using with model compilation
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # String identifier
    metrics=['mae']
)

# Using loss object directly
mse = dl.losses.MeanSquaredError()
model.compile(optimizer='adam', loss=mse)

# Manual computation
y_true = np.array([[2.0], [1.0], [3.0]])
y_pred = np.array([[2.5], [1.2], [2.8]])
loss_value = mse(y_true, y_pred)
print(f"MSE Loss: {loss_value}")  # MSE Loss: 0.043333
```

### `MeanAbsoluteError`

Mean absolute error loss for regression tasks.

#### Constructor

```python
MeanAbsoluteError(reduction='auto', name='mean_absolute_error')
```

**Mathematical Formula:**
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Example:**
```python
# L1 loss (robust to outliers)
mae = dl.losses.MeanAbsoluteError()

# Use with model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Custom usage
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.8])
loss = mae(y_true, y_pred)
```

### `Huber`

Huber loss combines MSE and MAE, robust to outliers.

#### Constructor

```python
Huber(delta=1.0, reduction='auto', name='huber_loss')
```

**Parameters:**
- `delta` (float, default=1.0): Threshold for switching between MSE and MAE.

**Mathematical Formula:**
```
Huber = { 0.5 * (y_true - y_pred)²           if |y_true - y_pred| ≤ δ
        { δ * (|y_true - y_pred| - 0.5 * δ)  otherwise
```

**Example:**
```python
# Huber loss with custom delta
huber = dl.losses.Huber(delta=1.5)
model.compile(optimizer='adam', loss=huber)
```

### `LogCosh`

Logarithm of the hyperbolic cosine loss.

**Example:**
```python
logcosh = dl.losses.LogCosh()
model.compile(optimizer='adam', loss='log_cosh')
```

## Classification Losses

### `CategoricalCrossentropy`

Categorical crossentropy for multi-class classification.

#### Constructor

```python
CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, 
                       reduction='auto', name='categorical_crossentropy')
```

**Parameters:**
- `from_logits` (bool, default=False): Whether predictions are logits or probabilities.
- `label_smoothing` (float, default=0.0): Label smoothing factor (0.0 to 1.0).
- `reduction` (str, default='auto'): Reduction method.

**Mathematical Formula:**
```
CCE = -Σ(y_true * log(y_pred))
```

**Example:**
```python
# Standard categorical crossentropy
cce = dl.losses.CategoricalCrossentropy()

# With label smoothing (reduces overfitting)
cce_smooth = dl.losses.CategoricalCrossentropy(label_smoothing=0.1)

# From logits (numerically more stable)
cce_logits = dl.losses.CategoricalCrossentropy(from_logits=True)

# Usage with model
model.compile(
    optimizer='adam',
    loss=cce_smooth,
    metrics=['accuracy']
)

# Manual example
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # One-hot encoded
y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]])
loss = cce(y_true, y_pred)
```

### `SparseCategoricalCrossentropy`

Categorical crossentropy with integer labels.

#### Constructor

```python
SparseCategoricalCrossentropy(from_logits=False, reduction='auto', 
                             name='sparse_categorical_crossentropy')
```

**Example:**
```python
# For integer labels (not one-hot encoded)
scce = dl.losses.SparseCategoricalCrossentropy()

# Usage
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Example with integer labels
y_true = np.array([1, 0, 2])  # Class indices
y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]])
loss = scce(y_true, y_pred)
```

### `BinaryCrossentropy`

Binary crossentropy for binary classification.

#### Constructor

```python
BinaryCrossentropy(from_logits=False, label_smoothing=0.0, 
                  reduction='auto', name='binary_crossentropy')
```

**Mathematical Formula:**
```
BCE = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```

**Example:**
```python
# Binary classification
bce = dl.losses.BinaryCrossentropy()

# With label smoothing
bce_smooth = dl.losses.BinaryCrossentropy(label_smoothing=0.1)

# Usage
model.compile(optimizer='adam', loss='binary_crossentropy')

# Example
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
loss = bce(y_true, y_pred)
```

### `Hinge`

Hinge loss for "maximum-margin" classification.

#### Constructor

```python
Hinge(reduction='auto', name='hinge')
```

**Mathematical Formula:**
```
Hinge = max(0, 1 - y_true * y_pred)
```

**Example:**
```python
# For SVM-like classification
hinge = dl.losses.Hinge()
model.compile(optimizer='adam', loss='hinge')
```

### `SquaredHinge`

Squared hinge loss.

**Example:**
```python
squared_hinge = dl.losses.SquaredHinge()
```

## Advanced Losses

### `FocalLoss`

Focal loss for addressing class imbalance.

#### Constructor

```python
FocalLoss(alpha=0.25, gamma=2.0, from_logits=False, reduction='auto', 
          name='focal_loss')
```

**Parameters:**
- `alpha` (float, default=0.25): Weighting factor for rare class.
- `gamma` (float, default=2.0): Focusing parameter.
- `from_logits` (bool, default=False): Whether predictions are logits.

**Mathematical Formula:**
```
FL = -α * (1 - p)^γ * log(p)
```

**Example:**
```python
# For imbalanced datasets
focal = dl.losses.FocalLoss(alpha=0.25, gamma=2.0)

# Custom focal loss for specific problem
focal_custom = dl.losses.FocalLoss(alpha=0.75, gamma=1.5)

model.compile(optimizer='adam', loss=focal)
```

### `TripletLoss`

Triplet loss for metric learning.

#### Constructor

```python
TripletLoss(margin=1.0, reduction='auto', name='triplet_loss')
```

**Parameters:**
- `margin` (float, default=1.0): Margin for triplet separation.

**Mathematical Formula:**
```
TL = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

**Example:**
```python
# For face recognition, image retrieval
triplet = dl.losses.TripletLoss(margin=0.5)

# Custom training loop for triplet loss
def triplet_training_step(model, anchor, positive, negative):
    with dl.GradientTape() as tape:
        anchor_emb = model(anchor, training=True)
        positive_emb = model(positive, training=True)
        negative_emb = model(negative, training=True)
        
        loss = triplet([anchor_emb, positive_emb, negative_emb])
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### `CosineSimilarity`

Cosine similarity loss.

#### Constructor

```python
CosineSimilarity(axis=-1, reduction='auto', name='cosine_similarity')
```

**Example:**
```python
# For similarity learning
cosine = dl.losses.CosineSimilarity()
model.compile(optimizer='adam', loss=cosine)
```

## Custom Loss Functions

### Creating Custom Losses

```python
def custom_mse_with_regularization(y_true, y_pred, reg_weight=0.01):
    """Custom MSE with L2 regularization on predictions."""
    mse = dl.backend.mean(dl.backend.square(y_true - y_pred))
    l2_reg = reg_weight * dl.backend.mean(dl.backend.square(y_pred))
    return mse + l2_reg

# Use as function
model.compile(optimizer='adam', loss=custom_mse_with_regularization)

# As a class
class CustomMSELoss(dl.losses.Loss):
    def __init__(self, reg_weight=0.01, name='custom_mse'):
        super().__init__(name=name)
        self.reg_weight = reg_weight
    
    def call(self, y_true, y_pred):
        mse = dl.backend.mean(dl.backend.square(y_true - y_pred))
        l2_reg = self.reg_weight * dl.backend.mean(dl.backend.square(y_pred))
        return mse + l2_reg
    
    def get_config(self):
        config = super().get_config()
        config.update({'reg_weight': self.reg_weight})
        return config

# Usage
custom_loss = CustomMSELoss(reg_weight=0.05)
model.compile(optimizer='adam', loss=custom_loss)
```

### Weighted Loss

```python
def weighted_categorical_crossentropy(class_weights):
    """Weighted categorical crossentropy for imbalanced classes."""
    def loss(y_true, y_pred):
        # Apply class weights
        weights = dl.backend.sum(class_weights * y_true, axis=-1)
        
        # Compute crossentropy
        cce = -dl.backend.sum(y_true * dl.backend.log(y_pred + 1e-7), axis=-1)
        
        # Apply weights
        return weights * cce
    
    return loss

# Usage for imbalanced dataset
class_weights = np.array([0.5, 2.0, 1.5])  # Weights for each class
weighted_loss = weighted_categorical_crossentropy(class_weights)
model.compile(optimizer='adam', loss=weighted_loss)
```

### Contrastive Loss

```python
class ContrastiveLoss(dl.losses.Loss):
    """Contrastive loss for siamese networks."""
    
    def __init__(self, margin=1.0, name='contrastive_loss'):
        super().__init__(name=name)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """
        y_true: 1 if same class, 0 if different
        y_pred: distance between embeddings
        """
        # Same class: minimize distance
        same_class_loss = y_true * dl.backend.square(y_pred)
        
        # Different class: maximize distance up to margin
        diff_class_loss = (1 - y_true) * dl.backend.square(
            dl.backend.maximum(0.0, self.margin - y_pred)
        )
        
        return 0.5 * (same_class_loss + diff_class_loss)

# Usage
contrastive = ContrastiveLoss(margin=2.0)
```

## Loss Combination

### Multi-task Learning

```python
def multi_task_loss(task1_weight=1.0, task2_weight=1.0):
    """Combined loss for multi-task learning."""
    def loss(y_true, y_pred):
        # Split predictions for different tasks
        task1_pred = y_pred[:, :10]  # First 10 outputs
        task2_pred = y_pred[:, 10:]  # Remaining outputs
        
        # Split ground truth
        task1_true = y_true[:, :10]
        task2_true = y_true[:, 10:]
        
        # Compute individual losses
        task1_loss = dl.losses.categorical_crossentropy(task1_true, task1_pred)
        task2_loss = dl.losses.mean_squared_error(task2_true, task2_pred)
        
        # Combine with weights
        return task1_weight * task1_loss + task2_weight * task2_loss
    
    return loss

# Usage
multi_loss = multi_task_loss(task1_weight=1.0, task2_weight=0.5)
model.compile(optimizer='adam', loss=multi_loss)
```

## Complete Examples

### Image Classification with Focal Loss

```python
import deeplearning_zzf as dl

# Model for imbalanced image classification
model = dl.Model()
model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Flatten())
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# Use focal loss for class imbalance
focal_loss = dl.losses.FocalLoss(alpha=0.25, gamma=2.0)
model.compile(
    optimizer='adam',
    loss=focal_loss,
    metrics=['accuracy']
)

# Train model
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
```

### Regression with Custom Loss

```python
# Custom loss that penalizes overestimation more than underestimation
def asymmetric_mse(y_true, y_pred, overestimate_penalty=2.0):
    """Asymmetric MSE that penalizes overestimation."""
    error = y_pred - y_true
    
    # Different penalties for over/under estimation
    underestimate_loss = dl.backend.square(dl.backend.minimum(error, 0))
    overestimate_loss = overestimate_penalty * dl.backend.square(dl.backend.maximum(error, 0))
    
    return dl.backend.mean(underestimate_loss + overestimate_loss)

# Model for price prediction
model = dl.Model()
model.add_layer(dl.layers.Dense(128, activation='relu', input_shape=(features,)))
model.add_layer(dl.layers.Dropout(0.2))
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dense(1))  # Single output for regression

model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: asymmetric_mse(y_true, y_pred, 2.0),
    metrics=['mae']
)
```