### Project Description: Handwritten Digit Classification using MNIST Dataset

Ai/ Deep Learning Digit Classification Model 
Here is the Colab Link
https://colab.research.google.com/drive/1UHoNFC6OR0vgKEcHSplqkm1QTz-Fi6-v?usp=sharing


#### Overview
The objective of this project is to develop a model that accurately identifies handwritten digits using the MNIST dataset. This classic problem in computer vision serves as an excellent introduction to understanding the workings of Convolutional Neural Networks (CNNs) and other deep learning architectures. We developed and compared two models: one using only Dense layers and another incorporating CNN layers with MaxPooling and Dropout layers.

#### Dataset
The MNIST dataset comprises 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is divided into 60,000 training images and 10,000 testing images.

#### Models

##### 1. Fully Connected Neural Network (Dense Layers)

**Architecture:**

- **Dense Layer:** 512 units, ReLU activation
- **Dense Layer:** 10 units, Softmax activation

**Model Summary:**

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten_4 (Flatten)         (None, 784)               0         
                                                                 
 dense_10 (Dense)            (None, 512)               401920    
                                                                 
 dropout_6 (Dropout)         (None, 512)               0         
                                                                 
 dense_11 (Dense)            (None, 512)               262656    
                                                                 
 dropout_7 (Dropout)         (None, 512)               0         
                                                                 
 dense_12 (Dense)            (None, 10)                5130      
                                                                 
=================================================================
Total params: 669706 (2.55 MB)
Trainable params: 669706 (2.55 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

**Performance:**

- **Training Accuracy:** 97.91%
- **Validation Accuracy:** 97.72%
- **Testing Accuracy:** 97.86%

**Training Loss:** 0.0066
**Validation Loss:** 0.0813

##### 2. Convolutional Neural Network (CNN)

**Architecture:**

- **Conv2D Layer:** 32 filters, kernel size 3x3, ReLU activation
- **MaxPooling2D Layer:** pool size 2x2
- **Conv2D Layer:** 64 filters, kernel size 3x3, ReLU activation
- **MaxPooling2D Layer:** pool size 2x2
- **Flatten Layer**
- **Dense Layer:** 128 units, ReLU activation
- **Dropout Layer:** 0.5 dropout rate
- **Dense Layer:** 10 units, Softmax activation

**Model Summary:**

```
Model: "sequential_9"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_13 (Conv2D)          (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d_12 (MaxPooli  (None, 13, 13, 32)        0         
 ng2D)                                                           
                                                                 
 conv2d_14 (Conv2D)          (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_13 (MaxPooli  (None, 5, 5, 64)          0         
 ng2D)                                                           
                                                                 
 flatten_7 (Flatten)         (None, 1600)              0         
                                                                 
 dense_17 (Dense)            (None, 128)               204928    
                                                                 
 dropout_8 (Dropout)         (None, 128)               0         
                                                                 
 dense_18 (Dense)            (None, 10)                1290      
                                                                 
=================================================================
Total params: 225034 (879.04 KB)
Trainable params: 225034 (879.04 KB)
Non-trainable params: 0 (0.00 Byte)
```

**Performance:**

- **Training Accuracy:** 99.30%
- **Validation Accuracy:** 99.00%
- **Testing Accuracy:** 99.26%

**Training Loss:** 0.0223
**Validation Loss:** 0.0353

**F1 Score:** 0.992595844333099

Confusion Matrix:
[[ 975    1    0    0    0    0    2    1    1    0]
 [   0 1133    1    1    0    0    0    0    0    0]
 [   0    0 1030    0    1    0    0    1    0    0]
 [   0    0    1 1004    0    4    0    0    1    0]
 [   0    0    0    0  981    0    0    0    0    1]
 [   2    0    0    6    0  880    1    1    1    1]
 [   2    2    0    0    3    4  947    0    0    0]
 [   0    3    5    0    0    0    0 1016    1    3]
 [   2    0    1    2    0    0    0    2  965    2]
 [   1    1    0    0    5    4    0    2    1  995]]
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       980
           1       0.99      1.00      1.00      1135
           2       0.99      1.00      1.00      1032
           3       0.99      0.99      0.99      1010
           4       0.99      1.00      0.99       982
           5       0.99      0.99      0.99       892
           6       1.00      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.99      0.99      0.99       974
           9       0.99      0.99      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000


#### Conclusion
Both models demonstrated high accuracy in classifying handwritten digits. The fully connected neural network achieved an impressive testing accuracy of 98.25%, while the CNN model outperformed it with a testing accuracy of 99.29%. The CNN model's superior performance is attributed to its ability to capture spatial hierarchies in images, thanks to convolutional and pooling layers. This project highlights the effectiveness of CNNs in image classification tasks and serves as a foundational step towards more complex computer vision applications.
