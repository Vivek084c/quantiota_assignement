
# Custom Neural Network Implementation

## Overview
A custom implementation of a neural network using TensorFlow for binary classification on MNIST dataset. The network features an entropy-based approach with custom dense layers and activation functions, demonstrating both custom and TensorFlow-based solutions for even/odd digit classification.

## Directory Structure
```
.
├── custom_neural_netowrk_final.ipynb    # Main implementation notebook
├── requirements.txt                     # Project dependencies
└── README.md                           # Project documentation
```

## Features
- Custom Dense Layer Implementation
- Binary Cross-Entropy Loss Function
- Sigmoid Activation Layer
- GPU Acceleration Support
- Performance Benchmarking

## Requirements
```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `custom_neural_netowrk_final.ipynb`

3. Run all cells sequentially

## Implementation Details
- Input Layer: 784 neurons (28x28 MNIST images)
- Hidden Layer: Custom dense implementation
- Output Layer: Binary classification (Even/Odd)
- Training: Custom gradient tape implementation
- Optimization: Adam optimizer

## Performance
- Training accuracy(On custom neural network): ~89.03%
- Training accuracy(On pre-build neural network): ~86.26%


## License
MIT License

## Author
Vivek Choudhry
