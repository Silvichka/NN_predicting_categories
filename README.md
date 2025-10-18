# Neural Network from Scratch

A pure Python/NumPy implementation of a Multi-Layer Perceptron (MLP) for binary classification, built without using deep learning frameworks like TensorFlow or PyTorch.

## Project Overview

This project demonstrates a comprehensive understanding of neural network fundamentals by implementing backpropagation, gradient descent, and various activation functions from scratch. The implementation achieves **100% accuracy** on the City Types classification dataset.

## General Purpose Classification
This neural network implementation is designed to be dataset-agnostic and can be applied to any normalized dataset for categorical prediction tasks. The architecture is flexible and configurable:

- **Adaptable input layer:** Automatically adjusts to any number of features in your dataset

- **Configurable hidden layers:** Customize the number of layers and neurons per layer based on your problem complexity

- **Multi-class classification:** Supports binary and multi-class categorical prediction through the softmax output layer

- **Preprocessing requirements:** Works with any normalized numerical data (StandardScaler recommended)

To use this network on your own dataset, simply:

1. Normalize your features (e.g., using StandardScaler)
2. Encode categorical labels if necessary 
3. Initialize the MLP with appropriate input dimensions and desired architecture 
4. Train and evaluate on your data

The implementation has been tested on classification problems and demonstrates the core principles of neural networks that can be applied across various domains including medical diagnosis, customer segmentation, fraud detection, and more.

## Technical Implementation

### Core Components

- **Neuron Class** (`src/Neuron.py`): Individual neuron with configurable activation functions
- **Layer Class** (`src/Layer.py`): Fully connected layer with multiple neurons
- **MLP Class** (`src/MLP.py`): Complete multi-layer perceptron with training and evaluation capabilities
- **Helper Functions** (`src/helper_functions.py`): Activation functions and loss calculations

### Features

- ✅ Custom backpropagation implementation
- ✅ Multiple activation functions (ReLU, Sigmoid)
- ✅ Softmax output layer for classification
- ✅ Cross-entropy loss function
- ✅ He and Xavier weight initialization
- ✅ Training visualization with loss and accuracy plots
- ✅ Comprehensive model evaluation with per-class metrics

## Architecture

```
Input Layer → Hidden Layer (8 neurons, ReLU) → Output Layer (2 neurons, Softmax)
```

## Technologies Used

- **Python 3.x**
- **NumPy**: Matrix operations and numerical computations
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Training visualization
- **scikit-learn**: Data preprocessing (StandardScaler, train_test_split)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:Silvichka/NN_predicting_categories.git
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### Basic Example

```python
from src.MLP import MLP
import numpy as np

# Initialize the network
# nin: number of input features
# nouts: list of neurons per layer [hidden_layer_size, output_size]
mlp = MLP(nin=10, nouts=[8, 2], activation='relu')

# Train the model
mlp.train_model(
    x_train, 
    y_train, 
    epochs=50, 
    print_report=True, 
    print_graph=True
)

# Evaluate on test set
mlp.evaluate_model(x_test, y_test, print_report=True)
```

### Running the Demo

```bash
jupyter notebook test_file.ipynb
```

The demo notebook includes:
- Data loading and preprocessing
- Model training with visualization
- Comprehensive evaluation metrics

## Results

On the City Types dataset:
- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- **Test Loss**: ~0.0000

The model successfully classifies cities into Residential and Industrial categories based on various features including temporal patterns (date, month, day, weekday) and numerical attributes.

## Project Structure

```
neural-network-from-scratch/
├── src/
│   ├── Neuron.py          # Individual neuron implementation
│   ├── Layer.py           # Layer of neurons
│   ├── MLP.py            # Multi-layer perceptron
│   └── helper_functions.py # Activation and loss functions
├── data/
│   └── City_Types.csv    # Dataset
├── test_file.ipynb       # Demo notebook
└── README.md
```
---

**Note**: This is an educational implementation. For production use cases, established frameworks like PyTorch or TensorFlow are recommended for their optimization and scalability.