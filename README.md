# Understanding and Estimating Error Propagation in Neural Networks for Scientific Data Analysis

## Overview

This repository contains the code and implementation details for our paper:

**"Understanding and Estimating Error Propagation in Neural Networks for Scientific Data Analysis"**

The work introduces a framework for optimizing neural network inference in scientific computing by combining data reduction and weight quantization, while maintaining error-controlled outcomes.

Our approach is evaluated on learning-based combustion simulations and satellite image classification, demonstrating a balance between computational performance and numerical precision.

## Running the Experiments

This section provides step-by-step instructions for reproducing our results.

### **1️ Data Preparation**

Before running experiments, ensure that you have the necessary datasets. The full combustion data is property, and access can only be granted by PI.

### **2️ Training the Model**

To train the model with default settings, run:

```bash
python [task]/train.py --input path_to_input_tensor.pth --target path_to_target_tensor.pth --checkpoint path_to_checkpoint.pth
```

For custom configurations, modify the `config/` directory.

### **3 Quantizing the Model**

To acquire quantized models of various formats:

```bash
python [task]/quantization.py --checkpoint path_to_checkpoint.pth --quantized path_to_output_folder
```

### **4 Evaluating the Model**

To evaluate a trained model:

```bash
python [task]/evaluate.py --input path_to_input_tensor.pth --reduced path_to_reduced_tensor.pth --checkpoint path_to_checkpoint.pth
```
