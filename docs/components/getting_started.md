# 快速入门指南

## 欢迎使用深度学习ZZF

本指南将帮助您快速上手深度学习ZZF框架，从安装到构建您的第一个神经网络模型。

## 安装

### 使用pip安装

```bash
pip install deeplearning-zzf
```

### 从源码安装

```bash
git clone https://github.com/gofenix/deeplearning.zzf.git
cd deeplearning.zzf
pip install -e .
```

### 验证安装

```python
import deeplearning_zzf as dl
print(f"深度学习ZZF版本: {dl.__version__}")
```

## 5分钟快速入门

### 1. 导入框架

```python
import deeplearning_zzf as dl
import numpy as np
```

### 2. 准备数据

```python
# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = dl.data.load_mnist()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = dl.utils.to_categorical(y_train, 10)
y_test = dl.utils.to_categorical(y_test, 10)
```

### 3. 构建模型

```python
# 创建序列模型
model = dl.Model()

# 添加层
model.add_layer(dl.layers.Flatten(input_shape=(28, 28)))
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.Dropout(0.2))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
```

### 4. 编译模型

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 5. 训练模型

```python
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_test, y_test)
)
```

### 6. 评估模型

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"测试准确率: {test_accuracy:.4f}")
```

## 核心概念

### 模型 (Model)

模型是神经网络的容器，它将层组织在一起。

```python
# 创建模型
model = dl.Model(name="my_model")

# 添加层
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
```

### 层 (Layers)

层是神经网络的基本构建块。

```python
# 全连接层
dense = dl.layers.Dense(64, activation='relu')

# 卷积层
conv = dl.layers.Conv2D(32, (3, 3), activation='relu')

# 循环层
lstm = dl.layers.LSTM(50, return_sequences=True)
```

### 优化器 (Optimizers)

优化器定义了如何更新模型权重。

```python
# Adam优化器
adam = dl.optimizers.Adam(learning_rate=0.001)

# SGD优化器
sgd = dl.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

### 损失函数 (Loss Functions)

损失函数衡量模型预测与真实值的差距。

```python
# 分类问题
categorical_loss = dl.losses.CategoricalCrossentropy()

# 回归问题
mse_loss = dl.losses.MeanSquaredError()
```

### 度量指标 (Metrics)

度量指标用于监控训练和验证过程。

```python
# 准确率
accuracy = dl.metrics.Accuracy()

# 精确率
precision = dl.metrics.Precision()
```

## 常见任务示例

### 图像分类

```python
# 构建CNN模型
model = dl.Model()

# 卷积层
model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu', 
                                 input_shape=(32, 32, 3)))
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
model.add_layer(dl.layers.MaxPooling2D((2, 2)))

# 分类器
model.add_layer(dl.layers.Flatten())
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# 编译
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

### 文本分类

```python
# 文本处理
vectorizer = dl.data.TextVectorization(max_tokens=10000, 
                                      output_sequence_length=100)

# 构建模型
model = dl.Model()
model.add_layer(vectorizer)
model.add_layer(dl.layers.Embedding(10000, 128))
model.add_layer(dl.layers.LSTM(64))
model.add_layer(dl.layers.Dense(1, activation='sigmoid'))

# 编译
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
```

### 回归任务

```python
# 构建回归模型
model = dl.Model()
model.add_layer(dl.layers.Dense(64, activation='relu', input_shape=(10,)))
model.add_layer(dl.layers.Dense(32, activation='relu'))
model.add_layer(dl.layers.Dense(1))  # 单个输出

# 编译
model.compile(optimizer='adam', 
              loss='mse', 
              metrics=['mae'])
```

## 数据处理

### 图像数据增强

```python
# 创建数据生成器
datagen = dl.data.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# 适配数据
datagen.fit(x_train)

# 生成增强数据
train_generator = datagen.flow(x_train, y_train, batch_size=32)
```

### 文本数据处理

```python
# 文本向量化
vectorizer = dl.data.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)

# 适配文本数据
vectorizer.adapt(text_data)

# 转换文本
sequences = vectorizer(text_data)
```

## 模型保存和加载

### 保存模型

```python
# 保存整个模型
model.save('my_model.h5')

# 只保存权重
model.save_weights('model_weights.h5')
```

### 加载模型

```python
# 加载整个模型
model = dl.load_model('my_model.h5')

# 加载权重
model.load_weights('model_weights.h5')
```

## 回调函数

回调函数提供了在训练过程中执行特定操作的机制。

```python
# 早停
early_stopping = dl.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True
)

# 学习率调度
lr_scheduler = dl.callbacks.ReduceLROnPlateau(
    factor=0.2, 
    patience=5
)

# 模型检查点
checkpoint = dl.callbacks.ModelCheckpoint(
    'best_model.h5', 
    save_best_only=True
)

# 使用回调
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, lr_scheduler, checkpoint],
    epochs=100
)
```

## 调试和优化技巧

### 可视化训练过程

```python
import matplotlib.pyplot as plt

# 绘制训练历史
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.legend()

plt.show()
```

### 模型摘要

```python
# 查看模型结构
model.summary()

# 可视化模型
dl.utils.plot_model(model, to_file='model.png', show_shapes=True)
```

### 学习率调优

```python
# 学习率查找器
def find_lr(model, train_data, start_lr=1e-7, end_lr=10, num_it=100):
    # 实现学习率查找器
    pass

# 使用不同的优化器
optimizers = {
    'adam': dl.optimizers.Adam(0.001),
    'sgd': dl.optimizers.SGD(0.01, momentum=0.9),
    'rmsprop': dl.optimizers.RMSprop(0.001)
}
```

## 下一步

- 查看[API参考文档](../api/)了解详细的函数和类说明
- 运行[示例代码](../../examples/)学习具体应用
- 阅读[高级指南](advanced_guide.md)了解更复杂的用法
- 参与[社区讨论](https://github.com/gofenix/deeplearning.zzf/discussions)

## 常见问题

### Q: 如何选择合适的学习率？

A: 通常从0.001开始，如果损失下降太慢，增大学习率；如果损失波动很大，减小学习率。

### Q: 什么时候使用不同的激活函数？

A: 
- ReLU：隐藏层的默认选择
- Sigmoid：二分类输出层
- Softmax：多分类输出层
- Tanh：RNN中常用

### Q: 如何防止过拟合？

A: 
- 使用Dropout层
- 添加正则化
- 增加训练数据
- 使用数据增强
- 早停训练

### Q: 模型训练很慢怎么办？

A: 
- 减小批次大小
- 使用GPU加速
- 简化模型结构
- 使用预训练模型

欢迎开始您的深度学习之旅！