# Flash Boilerplate
A general-purpose deep learning project template based on PyTorch Lightning.


## Core Philosophy
Many engineering tasks in deep learning projects are repetitive and tedious. To solve this, I've built this general-purpose template based on the core philosophy of **PyTorch Lightning** (**convention over configuration**).

While PyTorch Lightning already simplifies a lot of engineering development and training control, I've added extra conventions and encapsulations to further boost efficiency.

The core value of this template lies in:
- **Focusing on the core task:** You can concentrate on data processing, model design, and improving training methods without wasting time building a project framework from scratch.
- **Boosting development efficiency:** This template eliminates the repetitive setup phase of deep learning projects, allowing you to dive directly into model development.
- **Extensibility:** Despite its conventions, the template retains plenty of **hooks** to meet future customization and extension needs.

This repository is lightweight and powerful, with just over a thousand lines of pure code. It's an excellent starting point that you can clone and use as the foundation for your new projects.


## Quick Start
### Installation
```bash
git clone https://github.com/yuliu625/Yu-Flash-Boilerplate.git
cd Flash-Boilerplate
# Install all dependencies
# Refer to the dl_env configuration in the linked repository
```

### Project Structure
```bash
.
├── assets/                  # General assets, such as documentation
├── configs/                 # Configuration files
├── deep_learning_project/   # The main deep learning project
│   ├── torch_dataloaders/   # Data loaders and datasets
│   ├── torch_models/        # Model definitions
│   ├── trainer/             # Trainer-related utilities
│   ├── utils/               # General utility functions
│   └── schemas/             # Pydantic data schemas
├── scripts/                 # Run scripts
├── tests/                   # Tests
└── README.md
```

### Running
Use the example configuration files in the `configs` directory to run a training task directly:
```bash
bash scripts/run.sh
```


## Design Philosophy and Conventions
Based on my personal experience with various deep learning tasks, I've distilled a set of best practices and solidified them in this template. If you follow these conventions, you can maximize the template's benefits.

### 1\. Registration Mechanism: Configurable and Pluggable Components
All core components in the project are encapsulated using the **Factory Pattern** and **Strategy Pattern** and are managed through a registration mechanism. This means you can switch between different components by simply modifying the configuration file **without changing the code**, enabling complete deserialization of your training configurations.

Registrable components include:
- `torch_datasets`: Data loading methods.
- `torch_models`: Model definitions.
- `loss_fn`: The loss function that guides training.
- `torchmetrics_metrics`: Metrics for evaluating model performance.
- `callback`: Callbacks for training control.
- `logger`: Loggers.

For specific usage and build methods for these components, please refer to the detailed docstrings in the `__init__.py` file of each corresponding package.

### 2\. Data Processing Conventions
To ensure a consistent data flow, all data is passed using the `dict` format.
- **Dataset Output**: A `dict` returned by the `dataset` must contain the keys `'target'` and `'data'`.
- **Dataloader Output**: A `dict` returned by the `dataloader` must also contain the keys `'targets'` and `'datas'` (the `collate_fn` needs to be handled accordingly).

### 3\. Configuration Management
All project configurations are stored as `YAML` files in the `configs` directory.
- **Single configuration:** A single configuration file contains all settings for a task, making it easy to version control.
- **Batch generation:** The `jinja2` template engine can be used to generate configuration files in bulk, which is useful for large-scale experiments like hyperparameter searches.


## Use Cases
### This template is perfect for you if your needs are:
- **Deep learning network design:** You need to directly design and modify model architectures.
- **Multimodal tasks:** You need to customize intricate cross-modal computations.
- **Cutting-edge domain exploration:** Your field lacks existing toolkits, and you need to build from scratch.

### Alternatives
In some cases, there might be more suitable tools:
- **Keras:** For simple introductory trials or proof-of-concept projects, Keras is the easiest deep learning tool to get started with. (I don't like TensorFlow, though.)
- **Hugging Face Transformers:** If you're focusing on NLP tasks, the Transformers library offers a design philosophy similar to PyTorch Lightning with low-coupling components, making it a better choice. (I love it\!)


## Links and References
You can learn more about the tools and concepts that this project relies on through the following links:
- **General project structure:** [Path-Toolkit](https://github.com/yuliu625/Yu-Path-Toolkit)
- **Python environment configuration:** [Python-Environment-Configs](https://github.com/yuliu625/Yu-Python-Environment-Configs)
- **Deep learning toolkit:** [Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)

### Real-World Applications
Here are some real-world projects built on this template (or its predecessor) that you can use as a reference:

#### New Projects (focus on functional programming and design patterns):
- [Firm-Network-Predictor](https://github.com/yuliu625/Firm-Network-Predictor)

#### Older Projects (based on an earlier object-oriented design):
- [text-classification-trainer](https://github.com/yul1024/text-classification-trainer)
- [image-classification-trainer](https://github.com/yul1024/image-classification-trainer)
- [audio-classification-trainer](https://github.com/yul1024/audio-classification-trainer)


## Future Plans
I plan to further enhance this template in the future by:
  - **Integrating automated hyperparameter search:** Introducing tools like `optuna` or `ray[tune]` for more advanced automated experimentation.
  - **Adding distributed and parallel training support:** Leveraging `ray`'s capabilities to simplify large-scale data processing and distributed training.

