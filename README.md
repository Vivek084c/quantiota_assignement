# Custom Neural Network Implementation

## Overview
A custom implementation of a neural network for binary classification on MNIST dataset. The network features an entropy-based approach with custom dense layers and activation functions, demonstrating both custom and TensorFlow-based solutions for even/odd digit classification.

## implmentation of z mapping
We have added dual weight nature in the custom neural network  as defined below

```bash
self.weights = tf.Variable(tf.random.normal([num_inputs, num_neurons], stddev=0.01, dtype=tf.float32))
self.Gweights = tf.Variable(tf.random.normal([num_inputs, num_neurons], stddev=0.01, dtype=tf.float32))
```

The forward pass for the given neuron is defined as given below and weight updation constraint is given below

Forward pass: 
```bash
def forward(self, inputs, delta=0.5):
        self.inputs = inputs
        new_output = tf.matmul(inputs, self.weights) + tf.matmul(inputs, self.Gweights) + self.bias
```

Z update constration: 
```bash
if self.prev_output is not None:
            output_diff = tf.abs(new_output - self.prev_output)
            mask = tf.cast(output_diff < delta, dtype=tf.float32)
            self.output = mask * new_output + (1 - mask) * self.prev_output
        else:
            self.output = new_output  # First iteration, no previous output
```


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
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
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
