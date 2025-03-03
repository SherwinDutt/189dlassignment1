# dlassignment1
# Deep Learning Assignment 1

## Student Details
- **Roll No:** 189
- **Dataset Used:** CIFAR-10

## Project Overview
This project implements a **Feedforward Neural Network (FNN)** to classify images from the **CIFAR-10 dataset**. The model is trained using **backpropagation** and optimized with different techniques. The results are analyzed using accuracy metrics and confusion matrices.

---

## Dataset Information
- **CIFAR-10** consists of **60,000** 32x32 color images in **10 classes**:
  - Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Training Data:** 50,000 images  
- **Test Data:** 10,000 images

---

## Installation & Setup
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/deep-learning-assignment.git
cd deep-learning-assignment
```

### **2. Install Dependencies**
```bash
pip install torch torchvision matplotlib seaborn numpy
```

### **3. Run the Training Script**
```bash
python train.py
```

---

## Model Architecture
- **Input Layer:** 32x32x3 (Flattened to 3072 neurons)
- **Hidden Layers:** Configurable (Default: 128, 64 neurons)
- **Activation Function:** ReLU
- **Output Layer:** 10 neurons (Softmax)

---

## Training & Hyperparameters
| Hyperparameter       | Best Value     |
|----------------------|---------------|
| **Hidden Layers**    | 3 (128, 64, 32) |
| **Optimizer**        | Adam           |
| **Learning Rate**    | 0.001          |
| **Batch Size**       | 32             |
| **Activation Function** | ReLU        |

---

## Running Model Evaluation
```bash
python evaluate.py
```

### **Test Accuracy:** 79.1%

---

## Confusion Matrix
Run the following command to generate a confusion matrix:
```bash
python plot_confusion.py
```

---

## Comparing Loss Functions
| Loss Function | Final Accuracy |
|--------------|---------------|
| **Cross-Entropy Loss** | 79.1% |
| **Mean Squared Error (MSE) Loss** | 65.3% |

**Conclusion:** Cross-Entropy Loss performs significantly better than MSE for classification tasks.

---

## Key Takeaways
1. **Adam optimizer + Xavier initialization** provides the best performance.
2. **Batch size of 32** balances training speed and accuracy.
3. **ReLU activation** speeds up convergence.

---


