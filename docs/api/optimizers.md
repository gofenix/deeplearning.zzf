# Optimizers API Documentation

## Overview

Optimizers are algorithms that adjust the weights of neural networks to minimize the loss function. Deep Learning ZZF provides a variety of optimization algorithms for different use cases.

## Base Optimizer

### `Optimizer`

Abstract base class for all optimizers.

#### Methods

##### `apply_gradients(grads_and_vars)`

Apply gradients to variables.

**Parameters:**
- `grads_and_vars` (list): List of (gradient, variable) pairs.

##### `get_config()`

Returns the configuration of the optimizer as a dictionary.

## Gradient Descent Optimizers

### `SGD`

Stochastic Gradient Descent optimizer.

#### Constructor

```python
SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
```

**Parameters:**
- `learning_rate` (float, default=0.01): Learning rate.
- `momentum` (float, default=0.0): Momentum factor.
- `nesterov` (bool, default=False): Whether to apply Nesterov momentum.
- `name` (str, default='SGD'): Name for the optimizer.

**Example:**
```python
import deeplearning_zzf as dl

# Basic SGD
sgd = dl.optimizers.SGD(learning_rate=0.01)

# SGD with momentum
sgd_momentum = dl.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)

# SGD with Nesterov momentum
sgd_nesterov = dl.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

# Compile model with optimizer
model.compile(
    optimizer=sgd_momentum,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## Adaptive Learning Rate Optimizers

### `Adam`

Adaptive Moment Estimation optimizer.

#### Constructor

```python
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, 
     amsgrad=False, name='Adam')
```

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate.
- `beta_1` (float, default=0.9): Exponential decay rate for first moment estimates.
- `beta_2` (float, default=0.999): Exponential decay rate for second moment estimates.
- `epsilon` (float, default=1e-7): Small constant for numerical stability.
- `amsgrad` (bool, default=False): Whether to use AMSGrad variant.
- `name` (str, default='Adam'): Name for the optimizer.

**Example:**
```python
# Default Adam
adam = dl.optimizers.Adam()

# Custom Adam with different learning rate
adam_custom = dl.optimizers.Adam(learning_rate=0.0001)

# Adam with AMSGrad
adam_amsgrad = dl.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=True
)

# Use with model
model.compile(optimizer=adam_custom, loss='mse')
```

### `AdamW`

Adam with decoupled weight decay.

#### Constructor

```python
AdamW(learning_rate=0.001, weight_decay=0.01, beta_1=0.9, beta_2=0.999, 
      epsilon=1e-7, name='AdamW')
```

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate.
- `weight_decay` (float, default=0.01): Weight decay coefficient.
- `beta_1` (float, default=0.9): Exponential decay rate for first moment estimates.
- `beta_2` (float, default=0.999): Exponential decay rate for second moment estimates.
- `epsilon` (float, default=1e-7): Small constant for numerical stability.

**Example:**
```python
# AdamW with weight decay
adamw = dl.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.01
)
```

### `RMSprop`

RMSProp optimizer.

#### Constructor

```python
RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, 
        centered=False, name='RMSprop')
```

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate.
- `rho` (float, default=0.9): Discounting factor for the history/coming gradient.
- `momentum` (float, default=0.0): Momentum factor.
- `epsilon` (float, default=1e-7): Small constant for numerical stability.
- `centered` (bool, default=False): Whether to normalize by estimated variance.

**Example:**
```python
# Standard RMSprop
rmsprop = dl.optimizers.RMSprop()

# RMSprop with momentum
rmsprop_momentum = dl.optimizers.RMSprop(
    learning_rate=0.001,
    momentum=0.9
)

# Centered RMSprop
rmsprop_centered = dl.optimizers.RMSprop(
    learning_rate=0.001,
    centered=True
)
```

### `Adagrad`

Adagrad optimizer.

#### Constructor

```python
Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-7, 
        name='Adagrad')
```

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate.
- `initial_accumulator_value` (float, default=0.1): Initial value for accumulators.
- `epsilon` (float, default=1e-7): Small constant for numerical stability.

**Example:**
```python
adagrad = dl.optimizers.Adagrad(learning_rate=0.01)
```

### `Adadelta`

Adadelta optimizer.

#### Constructor

```python
Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-7, name='Adadelta')
```

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate.
- `rho` (float, default=0.95): Decay factor.
- `epsilon` (float, default=1e-7): Small constant for numerical stability.

**Example:**
```python
adadelta = dl.optimizers.Adadelta()
```

## Learning Rate Schedules

### `ExponentialDecay`

Exponential decay learning rate schedule.

#### Constructor

```python
ExponentialDecay(initial_learning_rate, decay_steps, decay_rate, 
                 staircase=False, name=None)
```

**Parameters:**
- `initial_learning_rate` (float): Initial learning rate.
- `decay_steps` (int): Number of steps after which to apply decay.
- `decay_rate` (float): Decay rate.
- `staircase` (bool, default=False): Whether to apply decay in discrete staircase.

**Example:**
```python
# Exponential decay schedule
lr_schedule = dl.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Use with optimizer
optimizer = dl.optimizers.Adam(learning_rate=lr_schedule)
```

### `CosineDecay`

Cosine decay learning rate schedule.

#### Constructor

```python
CosineDecay(initial_learning_rate, decay_steps, alpha=0.0, name=None)
```

**Parameters:**
- `initial_learning_rate` (float): Initial learning rate.
- `decay_steps` (int): Number of steps to decay over.
- `alpha` (float, default=0.0): Minimum learning rate as fraction of initial.

**Example:**
```python
cosine_schedule = dl.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    alpha=0.01
)
```

### `PolynomialDecay`

Polynomial decay learning rate schedule.

#### Constructor

```python
PolynomialDecay(initial_learning_rate, decay_steps, end_learning_rate=0.0001,
                power=1.0, cycle=False, name=None)
```

**Example:**
```python
poly_schedule = dl.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    end_learning_rate=0.01,
    power=0.5
)
```

### `PiecewiseConstantDecay`

Piecewise constant decay learning rate schedule.

#### Constructor

```python
PiecewiseConstantDecay(boundaries, values, name=None)
```

**Parameters:**
- `boundaries` (list): List of step boundaries.
- `values` (list): List of learning rate values.

**Example:**
```python
step_schedule = dl.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[1000, 2000],
    values=[0.1, 0.02, 0.005]
)
```

## Gradient Clipping

### Using with Optimizers

```python
# Clip gradients by norm
optimizer = dl.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)

# Clip gradients by value
optimizer = dl.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5
)

# Clip gradients by global norm
optimizer = dl.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=1.0
)
```

## Advanced Usage Examples

### Learning Rate Finder

```python
import deeplearning_zzf as dl
import numpy as np
import matplotlib.pyplot as plt

def find_learning_rate(model, train_data, start_lr=1e-7, end_lr=10, num_it=100):
    """Find optimal learning rate using learning rate range test."""
    
    # Store original weights
    original_weights = model.get_weights()
    
    # Create learning rate schedule
    lr_mult = (end_lr / start_lr) ** (1.0 / num_it)
    
    losses = []
    learning_rates = []
    
    # Set initial learning rate
    model.optimizer.learning_rate = start_lr
    
    for i in range(num_it):
        # Train on a batch
        batch_x, batch_y = next(train_data)
        loss = model.train_on_batch(batch_x, batch_y)
        
        # Store results
        losses.append(loss)
        learning_rates.append(model.optimizer.learning_rate.numpy())
        
        # Update learning rate
        model.optimizer.learning_rate *= lr_mult
        
        # Stop if loss explodes
        if i > 10 and loss > 4 * min(losses):
            break
    
    # Restore original weights
    model.set_weights(original_weights)
    
    return learning_rates, losses

# Usage
model = dl.Model()
# ... build model ...
model.compile(optimizer=dl.optimizers.Adam(), loss='mse')

# Find learning rate
lrs, losses = find_learning_rate(model, train_generator)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.show()
```

### Warm-up Learning Rate Schedule

```python
class WarmUpCosineDecay(dl.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=0.0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def __call__(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + dl.backend.cos(dl.backend.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

# Usage
warmup_schedule = WarmUpCosineDecay(
    warmup_steps=1000,
    total_steps=10000,
    max_lr=0.001,
    min_lr=1e-6
)

optimizer = dl.optimizers.Adam(learning_rate=warmup_schedule)
```

### Optimizer Comparison

```python
def compare_optimizers(model_fn, train_data, val_data, epochs=10):
    """Compare different optimizers on the same task."""
    
    optimizers = {
        'SGD': dl.optimizers.SGD(learning_rate=0.01),
        'SGD+Momentum': dl.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'Adam': dl.optimizers.Adam(learning_rate=0.001),
        'AdamW': dl.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
        'RMSprop': dl.optimizers.RMSprop(learning_rate=0.001),
    }
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"Training with {name}...")
        
        # Create fresh model
        model = model_fn()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            verbose=0
        )
        
        results[name] = history.history
    
    return results

# Usage
def create_model():
    model = dl.Model()
    model.add_layer(dl.layers.Dense(128, activation='relu', input_shape=(784,)))
    model.add_layer(dl.layers.Dropout(0.2))
    model.add_layer(dl.layers.Dense(10, activation='softmax'))
    return model

results = compare_optimizers(create_model, train_data, val_data)
```