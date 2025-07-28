# 深度学习ZZF - 完整API文档目录

## 📚 文档概览

这是深度学习ZZF框架的完整中文文档，包含了所有公共API、函数和组件的详细说明。

## 🚀 快速开始

- [快速入门指南](components/getting_started.md) - 5分钟上手深度学习ZZF
- [安装指南](components/installation.md) - 详细的安装说明
- [基础教程](components/basic_tutorial.md) - 从零开始学习

## 📖 API参考手册

### 核心API
- [核心API](api/core.md) - 模型、编译、训练等核心功能
  - Model类 - 神经网络模型容器
  - 模型编译和训练方法
  - 模型保存和加载
  - 工具函数

### 网络层
- [层API](api/layers.md) - 所有神经网络层的详细说明
  - 全连接层 (Dense)
  - 卷积层 (Conv2D, Conv1D)
  - 池化层 (MaxPooling, AveragePooling)
  - 循环层 (LSTM, GRU, SimpleRNN)
  - 标准化层 (BatchNormalization, LayerNormalization)
  - 正则化层 (Dropout)
  - 激活层 (Activation)
  - 工具层 (Flatten, Reshape, Lambda等)

### 优化器
- [优化器API](api/optimizers.md) - 各种优化算法
  - 梯度下降优化器 (SGD)
  - 自适应学习率优化器 (Adam, AdamW, RMSprop)
  - 学习率调度器
  - 梯度裁剪
  - 高级用法示例

### 损失函数
- [损失函数API](api/losses.md) - 各种损失函数
  - 回归损失 (MSE, MAE, Huber)
  - 分类损失 (CategoricalCrossentropy, BinaryCrossentropy)
  - 高级损失 (FocalLoss, TripletLoss)
  - 自定义损失函数
  - 多任务学习损失

### 度量指标
- [度量指标API](api/metrics.md) - 模型评估指标
  - 准确率指标 (Accuracy, BinaryAccuracy)
  - 回归指标 (MSE, MAE, R2Score)
  - 分类指标 (Precision, Recall, F1Score, AUC)
  - 高级指标 (MeanIoU, CosineSimilarity)
  - 自定义指标

### 数据处理
- [数据处理API](api/data.md) - 数据加载和预处理
  - 数据集类 (Dataset, TensorDataset)
  - 数据加载器 (DataLoader)
  - 图像数据增强 (ImageDataGenerator)
  - 文本处理 (TextVectorization)
  - 数据工具函数
  - 内置数据集

### 模型相关
- [模型API](api/models.md) - 预训练模型和模型工具
  - 预训练模型
  - 模型构建工具
  - 迁移学习
  - 模型可视化

### 工具函数
- [工具函数API](api/utils.md) - 实用工具函数
  - 数据处理工具
  - 模型工具
  - 可视化工具
  - 调试工具

## 🔧 组件文档

### 入门指南
- [快速入门](components/getting_started.md) - 快速上手指南
- [核心概念](components/core_concepts.md) - 框架核心概念解释
- [最佳实践](components/best_practices.md) - 使用最佳实践

### 高级指南
- [高级用法](components/advanced_guide.md) - 高级功能和技巧
- [自定义组件](components/custom_components.md) - 创建自定义层和函数
- [性能优化](components/performance.md) - 性能优化技巧
- [多GPU训练](components/multi_gpu.md) - 分布式训练

### 专题教程
- [计算机视觉](components/computer_vision.md) - 图像相关任务
- [自然语言处理](components/nlp.md) - 文本处理任务
- [时间序列](components/time_series.md) - 时间序列预测
- [强化学习](components/reinforcement_learning.md) - 强化学习应用

## 📝 示例代码

### 基础示例
- [基础分类](../examples/basic_classification.py) - MNIST手写数字识别
- [卷积神经网络](../examples/cnn_example.py) - CIFAR-10图像分类
- [循环神经网络](../examples/rnn_example.py) - 文本情感分析
- [回归任务](../examples/regression_example.py) - 房价预测

### 高级示例
- [迁移学习](../examples/transfer_learning.py) - 使用预训练模型
- [自定义训练循环](../examples/custom_training.py) - 自定义训练逻辑
- [生成对抗网络](../examples/gan_example.py) - GAN图像生成
- [自编码器](../examples/autoencoder_example.py) - 数据降维和重构

### 应用示例
- [目标检测](../examples/object_detection.py) - YOLO目标检测
- [图像分割](../examples/image_segmentation.py) - 语义分割
- [文本生成](../examples/text_generation.py) - RNN文本生成
- [推荐系统](../examples/recommendation.py) - 协同过滤

## 🛠️ 开发指南

### 贡献指南
- [如何贡献代码](../CONTRIBUTING.md) - 贡献代码的流程
- [代码规范](components/coding_standards.md) - 代码风格指南
- [测试指南](components/testing.md) - 编写和运行测试

### 发布说明
- [版本历史](components/changelog.md) - 版本更新记录
- [迁移指南](components/migration.md) - 版本升级指南
- [已知问题](components/known_issues.md) - 当前已知问题

## 🤔 常见问题

### 安装和配置
- GPU支持配置
- 依赖库冲突解决
- 不同操作系统安装问题

### 使用问题
- 模型训练慢的解决方案
- 内存溢出处理
- 精度问题调试

### 错误排查
- 常见错误信息解释
- 调试技巧
- 性能分析工具

## 📞 获取帮助

### 社区支持
- [GitHub Issues](https://github.com/gofenix/deeplearning.zzf/issues) - 报告Bug和功能请求
- [讨论区](https://github.com/gofenix/deeplearning.zzf/discussions) - 社区讨论
- [Stack Overflow](https://stackoverflow.com/questions/tagged/deeplearning-zzf) - 技术问答

### 学习资源
- [官方教程](components/tutorials.md) - 官方教程集合
- [视频教程](components/video_tutorials.md) - 视频学习资源
- [论文实现](components/paper_implementations.md) - 经典论文复现

## 🔍 API速查表

### 快速参考
```python
import deeplearning_zzf as dl

# 创建模型
model = dl.Model()

# 添加层
model.add_layer(dl.layers.Dense(64, activation='relu'))
model.add_layer(dl.layers.Dropout(0.2))
model.add_layer(dl.layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# 评估模型
model.evaluate(x_test, y_test)

# 预测
predictions = model.predict(x_new)
```

### 常用层类型
| 层类型 | 用途 | 示例 |
|--------|------|------|
| Dense | 全连接层 | `dl.layers.Dense(64, activation='relu')` |
| Conv2D | 二维卷积 | `dl.layers.Conv2D(32, (3,3))` |
| LSTM | 长短期记忆 | `dl.layers.LSTM(50)` |
| Dropout | 随机丢弃 | `dl.layers.Dropout(0.2)` |
| BatchNormalization | 批标准化 | `dl.layers.BatchNormalization()` |

### 常用优化器
| 优化器 | 适用场景 | 示例 |
|--------|----------|------|
| Adam | 通用默认选择 | `dl.optimizers.Adam(0.001)` |
| SGD | 需要动量时 | `dl.optimizers.SGD(0.01, momentum=0.9)` |
| RMSprop | RNN训练 | `dl.optimizers.RMSprop(0.001)` |

---

**注意**: 这是一个完整的API文档索引。点击任何链接查看详细的API说明、参数解释和使用示例。所有示例代码都经过测试，可以直接运行。

**更新日期**: 2024年