# Metrics API Documentation

## Overview

Metrics are used to monitor the training and testing steps of your model. They don't affect training (unlike losses) but provide valuable insights into model performance.

## Base Metric

### `Metric`

Abstract base class for all metrics.

#### Methods

##### `update_state(y_true, y_pred, sample_weight=None)`

Updates the metric state.

##### `result()`

Computes and returns the metric value.

##### `reset_state()`

Resets all metric state variables.

## Accuracy Metrics

### `Accuracy`

Calculates how often predictions match targets.

#### Constructor

```python
Accuracy(name='accuracy', dtype=None)
```

**Example:**
```python
import deeplearning_zzf as dl
import numpy as np

# Basic accuracy
accuracy = dl.metrics.Accuracy()

# Use with model compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']  # String identifier
)

# Manual computation
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([0, 2, 3, 4])
accuracy.update_state(y_true, y_pred)
print(f"Accuracy: {accuracy.result()}")  # 0.75

# Reset for new batch
accuracy.reset_state()
```

### `BinaryAccuracy`

Binary classification accuracy.

#### Constructor

```python
BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
```

**Parameters:**
- `threshold` (float, default=0.5): Decision threshold.

**Example:**
```python
binary_acc = dl.metrics.BinaryAccuracy(threshold=0.6)

# For binary classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[binary_acc]
)

# Manual example
y_true = np.array([1, 1, 0, 0])
y_pred = np.array([0.9, 0.4, 0.3, 0.1])  # Probabilities
binary_acc.update_state(y_true, y_pred)
print(f"Binary Accuracy: {binary_acc.result()}")
```

### `CategoricalAccuracy`

Categorical accuracy for multi-class classification.

**Example:**
```python
cat_acc = dl.metrics.CategoricalAccuracy()

# One-hot encoded labels
y_true = np.array([[0, 0, 1], [0, 1, 0]])
y_pred = np.array([[0.1, 0.1, 0.8], [0.05, 0.95, 0.0]])
cat_acc.update_state(y_true, y_pred)
```

### `SparseCategoricalAccuracy`

Accuracy for sparse categorical labels (integer labels).

**Example:**
```python
sparse_acc = dl.metrics.SparseCategoricalAccuracy()

# Integer labels
y_true = np.array([2, 1])
y_pred = np.array([[0.1, 0.1, 0.8], [0.05, 0.95, 0.0]])
sparse_acc.update_state(y_true, y_pred)
```

### `TopKCategoricalAccuracy`

Top-k categorical accuracy.

#### Constructor

```python
TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy', dtype=None)
```

**Parameters:**
- `k` (int, default=5): Number of top predictions to consider.

**Example:**
```python
# Top-5 accuracy for ImageNet-like problems
top5_acc = dl.metrics.TopKCategoricalAccuracy(k=5)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', top5_acc]
)
```

## Regression Metrics

### `MeanSquaredError`

Mean squared error metric.

**Example:**
```python
mse = dl.metrics.MeanSquaredError()

# For regression models
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[mse, 'mae']
)

# Manual computation
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.2, 2.9])
mse.update_state(y_true, y_pred)
print(f"MSE: {mse.result()}")
```

### `MeanAbsoluteError`

Mean absolute error metric.

**Example:**
```python
mae = dl.metrics.MeanAbsoluteError()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### `RootMeanSquaredError`

Root mean squared error metric.

**Example:**
```python
rmse = dl.metrics.RootMeanSquaredError()
```

### `MeanAbsolutePercentageError`

Mean absolute percentage error.

**Example:**
```python
mape = dl.metrics.MeanAbsolutePercentageError()

# Good for problems where relative error matters
model.compile(optimizer='adam', loss='mse', metrics=[mape])
```

### `R2Score`

Coefficient of determination (R²).

**Example:**
```python
r2 = dl.metrics.R2Score()

# Measures proportion of variance explained
model.compile(optimizer='adam', loss='mse', metrics=[r2])
```

## Classification Metrics

### `Precision`

Precision metric for classification.

#### Constructor

```python
Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
```

**Parameters:**
- `thresholds` (list, optional): Decision thresholds.
- `top_k` (int, optional): Consider top-k predictions.
- `class_id` (int, optional): Specific class to compute precision for.

**Example:**
```python
# Binary precision
precision = dl.metrics.Precision()

# Multi-class precision with thresholds
precision_thresh = dl.metrics.Precision(thresholds=[0.3, 0.5, 0.7])

# Class-specific precision
precision_class = dl.metrics.Precision(class_id=1)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[precision, 'recall', 'f1_score']
)
```

### `Recall`

Recall (sensitivity) metric.

#### Constructor

```python
Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
```

**Example:**
```python
recall = dl.metrics.Recall()

# For imbalanced datasets
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', recall]
)
```

### `F1Score`

F1 score (harmonic mean of precision and recall).

**Example:**
```python
f1 = dl.metrics.F1Score()

# Comprehensive classification metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', f1]
)
```

### `AUC`

Area under the ROC curve.

#### Constructor

```python
AUC(num_thresholds=200, curve='ROC', summation_method='interpolation',
    name=None, dtype=None, thresholds=None, multi_label=False, 
    num_labels=None, label_weights=None, from_logits=False)
```

**Parameters:**
- `num_thresholds` (int, default=200): Number of thresholds for curve approximation.
- `curve` (str, default='ROC'): Type of curve ('ROC' or 'PR').
- `multi_label` (bool, default=False): Whether this is multi-label classification.

**Example:**
```python
# ROC AUC
roc_auc = dl.metrics.AUC(name='roc_auc')

# Precision-Recall AUC (better for imbalanced datasets)
pr_auc = dl.metrics.AUC(curve='PR', name='pr_auc')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[roc_auc, pr_auc]
)
```

## Advanced Metrics

### `CosineSimilarity`

Cosine similarity between predictions and targets.

#### Constructor

```python
CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
```

**Example:**
```python
cosine_sim = dl.metrics.CosineSimilarity()

# For embedding similarity tasks
model.compile(
    optimizer='adam',
    loss='cosine_similarity',
    metrics=[cosine_sim]
)
```

### `MeanIoU`

Mean Intersection over Union for semantic segmentation.

#### Constructor

```python
MeanIoU(num_classes, name=None, dtype=None, ignore_class=None, 
        sparse_y_true=True, sparse_y_pred=True, axis=-1)
```

**Parameters:**
- `num_classes` (int): Number of classes.
- `ignore_class` (int, optional): Class to ignore in computation.
- `sparse_y_true` (bool, default=True): Whether y_true is sparse.
- `sparse_y_pred` (bool, default=True): Whether y_pred is sparse.

**Example:**
```python
# For semantic segmentation with 21 classes (Pascal VOC)
mean_iou = dl.metrics.MeanIoU(num_classes=21)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[mean_iou]
)
```

### `SparseCategoricalCrossentropy`

Sparse categorical crossentropy as a metric.

**Example:**
```python
scce_metric = dl.metrics.SparseCategoricalCrossentropy()

# Monitor crossentropy during training
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', scce_metric]
)
```

## Custom Metrics

### Creating Custom Metrics

```python
class F1ScoreCustom(dl.metrics.Metric):
    """Custom F1 score implementation."""
    
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision_metric = dl.metrics.Precision()
        self.recall_metric = dl.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        return 2 * (precision * recall) / (precision + recall + 1e-7)
    
    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()

# Usage
f1_custom = F1ScoreCustom()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_custom])
```

### Stateful Custom Metric

```python
class MeanAbsolutePercentageErrorCustom(dl.metrics.Metric):
    """Custom MAPE implementation with state tracking."""
    
    def __init__(self, name='mape', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize state variables
        self.total_percentage_error = self.add_weight(
            name='total_percentage_error', initializer='zeros'
        )
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = dl.backend.cast(y_true, self.dtype)
        y_pred = dl.backend.cast(y_pred, self.dtype)
        
        percentage_error = dl.backend.abs((y_true - y_pred) / (y_true + 1e-7)) * 100
        
        if sample_weight is not None:
            percentage_error *= sample_weight
        
        self.total_percentage_error.assign_add(dl.backend.sum(percentage_error))
        self.count.assign_add(dl.backend.size(y_true, out_type=self.dtype))
    
    def result(self):
        return dl.backend.divide_no_nan(self.total_percentage_error, self.count)
    
    def reset_state(self):
        self.total_percentage_error.assign(0.0)
        self.count.assign(0.0)
```

### Multi-output Metrics

```python
def multi_output_accuracy():
    """Custom metric for multi-output models."""
    def accuracy(y_true, y_pred):
        # Split outputs
        output1_true = y_true[:, :10]
        output1_pred = y_pred[:, :10]
        output2_true = y_true[:, 10:]
        output2_pred = y_pred[:, 10:]
        
        # Compute accuracies
        acc1 = dl.backend.mean(
            dl.backend.cast(
                dl.backend.equal(
                    dl.backend.argmax(output1_true, axis=-1),
                    dl.backend.argmax(output1_pred, axis=-1)
                ), 
                'float32'
            )
        )
        
        acc2 = dl.backend.mean(
            dl.backend.cast(
                dl.backend.equal(
                    dl.backend.argmax(output2_true, axis=-1),
                    dl.backend.argmax(output2_pred, axis=-1)
                ), 
                'float32'
            )
        )
        
        return (acc1 + acc2) / 2
    
    return accuracy

# Usage
multi_acc = multi_output_accuracy()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[multi_acc])
```

## Metric Utilities

### Confusion Matrix

```python
def confusion_matrix_metric(num_classes):
    """Custom confusion matrix metric."""
    def confusion_matrix(y_true, y_pred):
        # Convert to class indices
        y_true_indices = dl.backend.argmax(y_true, axis=-1)
        y_pred_indices = dl.backend.argmax(y_pred, axis=-1)
        
        # Create confusion matrix
        cm = dl.backend.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                mask = dl.backend.logical_and(
                    dl.backend.equal(y_true_indices, i),
                    dl.backend.equal(y_pred_indices, j)
                )
                cm = dl.backend.tensor_scatter_nd_add(
                    cm, [[i, j]], [dl.backend.sum(dl.backend.cast(mask, 'float32'))]
                )
        
        return cm
    
    return confusion_matrix
```

### Class-wise Metrics

```python
class ClasswiseAccuracy(dl.metrics.Metric):
    """Accuracy computed for each class separately."""
    
    def __init__(self, num_classes, name='classwise_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Create state variables for each class
        self.correct_per_class = []
        self.total_per_class = []
        
        for i in range(num_classes):
            self.correct_per_class.append(
                self.add_weight(f'correct_class_{i}', initializer='zeros')
            )
            self.total_per_class.append(
                self.add_weight(f'total_class_{i}', initializer='zeros')
            )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_classes = dl.backend.argmax(y_true, axis=-1)
        y_pred_classes = dl.backend.argmax(y_pred, axis=-1)
        
        for class_id in range(self.num_classes):
            # Mask for current class
            class_mask = dl.backend.equal(y_true_classes, class_id)
            
            # Correct predictions for this class
            correct_mask = dl.backend.logical_and(
                class_mask,
                dl.backend.equal(y_true_classes, y_pred_classes)
            )
            
            # Update counters
            self.correct_per_class[class_id].assign_add(
                dl.backend.sum(dl.backend.cast(correct_mask, 'float32'))
            )
            self.total_per_class[class_id].assign_add(
                dl.backend.sum(dl.backend.cast(class_mask, 'float32'))
            )
    
    def result(self):
        accuracies = []
        for i in range(self.num_classes):
            acc = dl.backend.divide_no_nan(
                self.correct_per_class[i], 
                self.total_per_class[i]
            )
            accuracies.append(acc)
        
        return dl.backend.stack(accuracies)
    
    def reset_state(self):
        for i in range(self.num_classes):
            self.correct_per_class[i].assign(0.0)
            self.total_per_class[i].assign(0.0)
```

## Complete Examples

### Binary Classification with Comprehensive Metrics

```python
import deeplearning_zzf as dl

# Build model
model = dl.Model()
model.add_layer(dl.layers.Dense(64, activation='relu', input_shape=(20,)))
model.add_layer(dl.layers.Dropout(0.5))
model.add_layer(dl.layers.Dense(32, activation='relu'))
model.add_layer(dl.layers.Dense(1, activation='sigmoid'))

# Comprehensive metrics for binary classification
metrics = [
    'accuracy',
    dl.metrics.Precision(),
    dl.metrics.Recall(),
    dl.metrics.F1Score(),
    dl.metrics.AUC(name='roc_auc'),
    dl.metrics.AUC(curve='PR', name='pr_auc')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=metrics
)

# Train with validation metrics
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate on test set
test_results = model.evaluate(x_test, y_test, verbose=0)
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")
```

### Multi-class Classification with Top-K Accuracy

```python
# Model for ImageNet-like classification
model = dl.Model()
# ... build CNN architecture ...
model.add_layer(dl.layers.Dense(1000, activation='softmax'))  # 1000 classes

# Multiple accuracy metrics
metrics = [
    'accuracy',  # Top-1 accuracy
    dl.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
    dl.metrics.TopKCategoricalAccuracy(k=10, name='top_10_accuracy'),
    dl.metrics.SparseCategoricalCrossentropy(name='crossentropy')
]

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=metrics
)
```