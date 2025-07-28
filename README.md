# Deep Learning ZZF

A comprehensive deep learning framework and toolkit.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Components](#components)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
pip install deeplearning-zzf
```

### Development Installation

```bash
git clone https://github.com/gofenix/deeplearning.zzf.git
cd deeplearning.zzf
pip install -e .
```

## Quick Start

```python
import deeplearning_zzf as dl

# Basic usage example
model = dl.Model()
model.add_layer(dl.layers.Dense(128, activation='relu'))
model.add_layer(dl.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

## API Reference

Detailed API documentation is available in the [docs/api](docs/api) directory.

- [Core APIs](docs/api/core.md)
- [Models](docs/api/models.md)
- [Layers](docs/api/layers.md)
- [Optimizers](docs/api/optimizers.md)
- [Loss Functions](docs/api/losses.md)
- [Metrics](docs/api/metrics.md)
- [Data Processing](docs/api/data.md)
- [Utilities](docs/api/utils.md)

## Components

See [Component Documentation](docs/components) for detailed information about each component.

## Examples

Check out the [examples](examples) directory for comprehensive usage examples:

- [Basic Classification](examples/basic_classification.py)
- [Convolutional Neural Networks](examples/cnn_example.py)
- [Recurrent Neural Networks](examples/rnn_example.py)
- [Transfer Learning](examples/transfer_learning.py)
- [Custom Training Loop](examples/custom_training.py)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
