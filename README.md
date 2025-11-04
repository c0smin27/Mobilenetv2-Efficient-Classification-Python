# Efficient Classification with Depthwise Separable Convolutions

## Description

This project explores efficient image classification using **MobileNetV2**, a lightweight convolutional neural network designed for **mobile and embedded devices**.  
The focus is on **speed vs. accuracy trade-offs**, using **depthwise separable convolutions** and **inverted residuals** to significantly reduce model size and computational cost.  
All experiments were conducted in **Google Colab** using the **Fashion-MNIST** dataset.

**❗❗ To view the complete notebook with code, outputs, and graphs, download the file `EfficientClassification_CosminM_Proiect.html` and open it locally in any web browser. ❗❗**

## Objective

Classify grayscale clothing images (28×28 pixels) into **10 categories** using a compact MobileNetV2 architecture optimized for efficiency.

## How the Model Works

1. **Dataset**: Fashion-MNIST, split into 54,000 training, 6,000 validation, and 10,000 test images.  
2. **Model Architecture**: MobileNetV2 pretrained on ImageNet, with width multiplier `α` and adjustable input resolution.  
3. **Training**: Fine-tuned using Adam optimizer (learning rate = 1e-3) and lightweight data augmentation (flip, rotation).  
4. **Evaluation**: Metrics include accuracy, F1-score, model parameters, and epoch time.  
5. **Optimization**: Compared multiple configurations for input size (96×96, 128×128) and α (1.0, 0.75) to identify the best trade-off.

## Key Results

| Input Size | Alpha | Parameters | Val Accuracy | Test Accuracy | Sec/Epoch | Epochs |
|-------------|--------|-------------|---------------|----------------|------------|---------|
| 96×96 | 1.00 | 2,270,794 | 0.9193 | 0.9191 | 1035 | 4 |
| 128×128 | 1.00 | 2,270,794 | 0.9195 | 0.9153 | 1883 | 4 |
| 128×128 | 0.75 | 1,394,874 | 0.9202 | 0.9132 | 1549 | 4 |

**Observations:**
- Resolution 96×96 is sufficient for Fashion-MNIST. Increasing to 128×128 adds computation time without accuracy gains.  
- Lowering α to 0.75 cuts parameters by ~40% with less than 1% accuracy loss.  
- The best configuration is **96×96, α=1.0**, offering high accuracy with minimal cost.

## Technologies Used

- Python (**Google Colab**)  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  

## Project File

- `EfficientClassification_CosminM_Proiect.html` - full notebook with implementation, training logs, and analysis.

## Conclusion

MobileNetV2 achieves excellent accuracy on Fashion-MNIST with minimal computational requirements.  
This confirms the advantage of **depthwise separable convolutions** for efficient deep learning models suited for real-time and low-power applications.

## Disclaimer

This project was developed for educational purposes. It can be used for research or learning but should not be submitted as original coursework.
