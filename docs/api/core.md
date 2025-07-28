# 核心API文档

## 概述

核心API提供了深度学习ZZF框架的基础构建块。

## 类

### `Model`

用于构建和训练神经网络的主要模型类。

#### 构造函数

```python
Model(name=None, dynamic=False)
```

**参数：**
- `name` (str, 可选): 模型名称。如果为None，将生成唯一名称。
- `dynamic` (bool, 默认False): 是否启用动态图执行。

**示例：**
```python
import deeplearning_zzf as dl

# 创建基础模型
model = dl.Model(name="my_classifier")

# 创建动态模型
dynamic_model = dl.Model(dynamic=True)
```

#### 方法

##### `add_layer(layer)`

向模型添加层。

**参数：**
- `layer` (Layer): 要添加到模型的层实例。

**返回：**
- `Model`: 返回self以支持方法链式调用。

**示例：**
```python
model = dl.Model()
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.Dropout(0.2))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
```

##### `compile(optimizer, loss, metrics=None)`

配置模型以进行训练。

**参数：**
- `optimizer` (str 或 Optimizer): 要使用的优化算法。
- `loss` (str 或 Loss): 要最小化的损失函数。
- `metrics` (list, 可选): 训练期间要监控的指标列表。

**示例：**
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)
```

##### `fit(x, y, batch_size=32, epochs=1, validation_data=None, callbacks=None)`

在给定数据上训练模型。

**参数：**
- `x` (array-like): 输入数据。
- `y` (array-like): 目标数据。
- `batch_size` (int, 默认32): 每次梯度更新的样本数。
- `epochs` (int, 默认1): 训练模型的轮数。
- `validation_data` (tuple, 可选): 用于评估损失和指标的数据。
- `callbacks` (list, 可选): 训练期间应用的回调函数列表。

**返回：**
- `History`: 训练历史对象。

**示例：**
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

为输入样本生成预测。

**参数：**
- `x` (array-like): 输入数据。
- `batch_size` (int, 可选): 预测时的批次大小。

**返回：**
- `array`: 预测结果。

**示例：**
```python
predictions = model.predict(x_test, batch_size=128)
```

##### `evaluate(x, y, batch_size=None)`

在测试数据上评估模型。

**参数：**
- `x` (array-like): 输入数据。
- `y` (array-like): 目标数据。
- `batch_size` (int, 可选): 评估时的批次大小。

**返回：**
- `dict`: 包含损失和指标值的字典。

**示例：**
```python
results = model.evaluate(x_test, y_test, batch_size=128)
print(f"测试准确率: {results['accuracy']:.4f}")
```

##### `save(filepath)`

将模型保存到文件。

**参数：**
- `filepath` (str): 保存模型的路径。

**示例：**
```python
model.save('my_model.h5')
```

##### `load(filepath)` (类方法)

从文件加载模型。

**参数：**
- `filepath` (str): 保存模型的路径。

**返回：**
- `Model`: 加载的模型实例。

**示例：**
```python
loaded_model = dl.Model.load('my_model.h5')
```

## 函数

### `load_model(filepath)`

加载保存模型的工具函数。

**参数：**
- `filepath` (str): 保存模型的路径。

**返回：**
- `Model`: 加载的模型实例。

**示例：**
```python
model = dl.load_model('trained_model.h5')
```

### `set_random_seed(seed)`

设置随机种子以获得可重现的结果。

**参数：**
- `seed` (int): 随机种子值。

**示例：**
```python
dl.set_random_seed(42)
```

## 常量

### `__version__`

包含框架版本号的字符串。

```python
print(f"深度学习 ZZF 版本: {dl.__version__}")
```