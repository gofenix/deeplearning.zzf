# 层API文档

## 概述

层是深度学习ZZF中神经网络的基本构建块。每个层都会转换输入数据并将结果传递给下一层。

## 基础层

### `Layer`

所有层的抽象基类。

#### 方法

##### `call(inputs)`

层的前向传播逻辑。

**参数：**
- `inputs` (Tensor): 输入张量。

**返回：**
- `Tensor`: 输出张量。

##### `build(input_shape)`

创建层的权重。

**参数：**
- `input_shape` (tuple): 输入的形状。

## 全连接层

### `Dense`

全连接层。

#### 构造函数

```python
Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
      activity_regularizer=None)
```

**参数：**
- `units` (int): 输出单元数。
- `activation` (str 或 callable, 可选): 要使用的激活函数。
- `use_bias` (bool, 默认True): 是否使用偏置权重。
- `kernel_initializer` (str 或 callable, 默认'glorot_uniform'): 核权重初始化器。
- `bias_initializer` (str 或 callable, 默认'zeros'): 偏置权重初始化器。
- `kernel_regularizer` (str 或 callable, 可选): 核权重正则化器。
- `bias_regularizer` (str 或 callable, 可选): 偏置权重正则化器。
- `activity_regularizer` (str 或 callable, 可选): 层激活正则化器。

**示例：**
```python
import deeplearning_zzf as dl

# 基础全连接层
dense1 = dl.layers.Dense(128, activation='relu')

# 带正则化的全连接层
dense2 = dl.layers.Dense(
    64, 
    activation='relu',
    kernel_regularizer=dl.regularizers.l2(0.01),
    bias_regularizer=dl.regularizers.l1(0.01)
)

# 分类输出层
output = dl.layers.Dense(10, activation='softmax')
```

## 卷积层

### `Conv2D`

二维卷积层。

#### 构造函数

```python
Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
       activation=None, use_bias=True, kernel_initializer='glorot_uniform',
       bias_initializer='zeros')
```

**参数：**
- `filters` (int): 输出滤波器数量。
- `kernel_size` (int 或 tuple): 卷积核大小。
- `strides` (int 或 tuple, 默认(1, 1)): 卷积步长。
- `padding` (str, 默认'valid'): 填充策略（'valid' 或 'same'）。
- `activation` (str 或 callable, 可选): 激活函数。
- `use_bias` (bool, 默认True): 是否使用偏置。
- `kernel_initializer` (str 或 callable, 默认'glorot_uniform'): 核初始化器。
- `bias_initializer` (str 或 callable, 默认'zeros'): 偏置初始化器。

**示例：**
```python
# 基础卷积层
conv1 = dl.layers.Conv2D(32, (3, 3), activation='relu')

# 带填充和步长的卷积层
conv2 = dl.layers.Conv2D(
    64, 
    kernel_size=(3, 3), 
    strides=(2, 2), 
    padding='same',
    activation='relu'
)

# 深度可分离卷积
conv3 = dl.layers.Conv2D(128, (1, 1), activation='relu')
```

### `Conv1D`

用于序列数据的一维卷积层。

#### 构造函数

```python
Conv1D(filters, kernel_size, strides=1, padding='valid', activation=None)
```

**示例：**
```python
# 用于序列处理
conv1d = dl.layers.Conv1D(64, 3, activation='relu', padding='same')
```

## 池化层

### `MaxPooling2D`

二维空间数据的最大池化操作。

#### 构造函数

```python
MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
```

**参数：**
- `pool_size` (int 或 tuple, 默认(2, 2)): 池化窗口大小。
- `strides` (int 或 tuple, 可选): 池化操作的步长。
- `padding` (str, 默认'valid'): 填充策略。

**示例：**
```python
# 标准最大池化
maxpool = dl.layers.MaxPooling2D((2, 2))

# 自定义步长的池化
maxpool_custom = dl.layers.MaxPooling2D(
    pool_size=(3, 3), 
    strides=(2, 2), 
    padding='same'
)
```

### `AveragePooling2D`

二维空间数据的平均池化操作。

**示例：**
```python
avgpool = dl.layers.AveragePooling2D((2, 2))
```

### `GlobalMaxPooling2D`

二维空间数据的全局最大池化。

**示例：**
```python
global_max = dl.layers.GlobalMaxPooling2D()
```

### `GlobalAveragePooling2D`

二维空间数据的全局平均池化。

**示例：**
```python
global_avg = dl.layers.GlobalAveragePooling2D()
```

## 循环层

### `LSTM`

长短期记忆层。

#### 构造函数

```python
LSTM(units, activation='tanh', recurrent_activation='sigmoid', 
     use_bias=True, dropout=0.0, recurrent_dropout=0.0, 
     return_sequences=False, return_state=False, stateful=False)
```

**参数：**
- `units` (int): LSTM单元数。
- `activation` (str, 默认'tanh'): 细胞状态的激活函数。
- `recurrent_activation` (str, 默认'sigmoid'): 循环步骤的激活函数。
- `use_bias` (bool, 默认True): 是否使用偏置。
- `dropout` (float, 默认0.0): 输入的丢弃率。
- `recurrent_dropout` (float, 默认0.0): 循环连接的丢弃率。
- `return_sequences` (bool, 默认False): 是否返回完整序列。
- `return_state` (bool, 默认False): 是否返回最后状态。
- `stateful` (bool, 默认False): 是否在批次间保持状态。

**示例：**
```python
# 基础LSTM
lstm1 = dl.layers.LSTM(128, return_sequences=True)

# 带丢弃的LSTM
lstm2 = dl.layers.LSTM(
    64, 
    dropout=0.2, 
    recurrent_dropout=0.2,
    return_sequences=False
)

# 双向LSTM
bilstm = dl.layers.Bidirectional(dl.layers.LSTM(128))
```

### `GRU`

门控循环单元层。

#### 构造函数

```python
GRU(units, activation='tanh', recurrent_activation='sigmoid',
    dropout=0.0, recurrent_dropout=0.0, return_sequences=False)
```

**示例：**
```python
gru = dl.layers.GRU(128, return_sequences=True, dropout=0.2)
```

### `SimpleRNN`

简单循环神经网络层。

**示例：**
```python
rnn = dl.layers.SimpleRNN(64, return_sequences=False)
```

## 标准化层

### `BatchNormalization`

批量标准化层。

#### 构造函数

```python
BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)
```

**参数：**
- `axis` (int, 默认-1): 要标准化的轴。
- `momentum` (float, 默认0.99): 移动平均的动量。
- `epsilon` (float, 默认1e-3): 数值稳定性的小浮点数。
- `center` (bool, 默认True): 是否添加偏移参数。
- `scale` (bool, 默认True): 是否添加缩放参数。

**示例：**
```python
# 标准批量标准化
bn = dl.layers.BatchNormalization()

# 自定义批量标准化
bn_custom = dl.layers.BatchNormalization(
    momentum=0.95, 
    epsilon=1e-5
)
```

### `LayerNormalization`

层标准化。

**示例：**
```python
ln = dl.layers.LayerNormalization()
```

## 正则化层

### `Dropout`

用于正则化的丢弃层。

#### 构造函数

```python
Dropout(rate, noise_shape=None, seed=None)
```

**参数：**
- `rate` (float): 要丢弃的输入比例（0到1之间）。
- `noise_shape` (tuple, 可选): 生成丢弃掩码的形状。
- `seed` (int, 可选): 随机种子。

**示例：**
```python
# 标准丢弃
dropout1 = dl.layers.Dropout(0.2)

# 自定义噪声形状的丢弃
dropout2 = dl.layers.Dropout(0.5, noise_shape=(None, 1, 1, 1))
```

## 激活层

### `Activation`

对输入应用激活函数。

#### 构造函数

```python
Activation(activation)
```

**参数：**
- `activation` (str 或 callable): 要应用的激活函数。

**示例：**
```python
# 常见激活函数
relu = dl.layers.Activation('relu')
sigmoid = dl.layers.Activation('sigmoid')
tanh = dl.layers.Activation('tanh')
softmax = dl.layers.Activation('softmax')

# 自定义激活函数
def custom_activation(x):
    return x * dl.backend.sigmoid(x)  # Swish激活

swish = dl.layers.Activation(custom_activation)
```

## 工具层

### `Flatten`

将输入张量展平为1D。

**示例：**
```python
flatten = dl.layers.Flatten()
```

### `Reshape`

将输入张量重塑为指定形状。

#### 构造函数

```python
Reshape(target_shape)
```

**参数：**
- `target_shape` (tuple): 目标形状（不包括批次维度）。

**示例：**
```python
# 重塑为2D
reshape = dl.layers.Reshape((28, 28))

# 重塑为3D
reshape_3d = dl.layers.Reshape((7, 7, 64))
```

### `Permute`

置换输入张量的维度。

#### 构造函数

```python
Permute(dims)
```

**参数：**
- `dims` (tuple): 置换模式。

**示例：**
```python
# 从(batch, height, width, channels)改为(batch, channels, height, width)
permute = dl.layers.Permute((3, 1, 2))
```

### `Lambda`

将任意表达式包装为层。

#### 构造函数

```python
Lambda(function, output_shape=None)
```

**参数：**
- `function` (callable): 要评估的函数。
- `output_shape` (tuple, 可选): 期望的输出形状。

**示例：**
```python
# 自定义操作
square = dl.layers.Lambda(lambda x: x ** 2)

# 带输出形状
normalize = dl.layers.Lambda(
    lambda x: x / dl.backend.sqrt(dl.backend.sum(x**2, axis=-1, keepdims=True)),
    output_shape=lambda input_shape: input_shape
)
```

## 完整示例

```python
import deeplearning_zzf as dl

# 构建用于图像分类的CNN
model = dl.Model()

# 特征提取
model.add_layer(dl.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Dropout(0.25))

model.add_layer(dl.layers.Conv2D(64, (3, 3), activation='relu'))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.MaxPooling2D((2, 2)))
model.add_layer(dl.layers.Dropout(0.25))

# 分类头
model.add_layer(dl.layers.Flatten())
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.BatchNormalization())
model.add_layer(dl.layers.Dropout(0.5))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# 编译和摘要
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```