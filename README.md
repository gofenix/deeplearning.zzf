# 深度学习 ZZF 框架

一个全面的深度学习框架和工具包。

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [组件](#组件)
- [示例](#示例)
- [贡献](#贡献)
- [许可证](#许可证)

## 安装

```bash
pip install deeplearning-zzf
```

### 开发环境安装

```bash
git clone https://github.com/gofenix/deeplearning.zzf.git
cd deeplearning.zzf
pip install -e .
```

## 快速开始

```python
import deeplearning_zzf as dl

# 基本使用示例
model = dl.Model()
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

## API 参考

详细的API文档位于 [docs/api](docs/api) 目录。

- [核心API](docs/api/core.md)
- [模型](docs/api/models.md)
- [层](docs/api/layers.md)
- [优化器](docs/api/optimizers.md)
- [损失函数](docs/api/losses.md)
- [度量指标](docs/api/metrics.md)
- [数据处理](docs/api/data.md)
- [工具函数](docs/api/utils.md)

## 组件

查看 [组件文档](docs/components) 了解每个组件的详细信息。

## 示例

查看 [examples](examples) 目录中的完整使用示例：

- [基础分类](examples/basic_classification.py)
- [卷积神经网络](examples/cnn_example.py)
- [循环神经网络](examples/rnn_example.py)
- [迁移学习](examples/transfer_learning.py)
- [自定义训练循环](examples/custom_training.py)

## 贡献

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解我们的行为准则和提交拉取请求的流程。

## 许可证

此项目采用 MIT 许可证 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。
