# CNN-Image-Classification-on-CIFAR-10-Dataset-Deep-Learning-Model-for-AI-Image-Recognition by @m-faizanmahmod


This project demonstrates a deep learning-based image classification model using Convolutional Neural Networks (CNN) to classify images from the CIFAR-10 dataset. The model is implemented with TensorFlow and Keras, leveraging advanced machine learning techniques in Python. The dataset consists of 60,000 32x32 color images categorized into 10 classes.

## Key Technologies

- **Machine Learning (ML)**
- **Deep Learning (DL)**
- **Convolutional Neural Networks (CNN)**
- **Computer Vision (CV)**
- **Keras** 
- **TensorFlow**
- **Python**
- **NumPy, Matplotlib, Pandas**

## Project Overview

This project utilizes **deep learning** and **computer vision** techniques to develop a **CNN model** for **image classification**. The CIFAR-10 dataset, which contains 60,000 images categorized into 10 different classes, is used for training and testing. The dataset is preprocessed, and the model is trained to predict image classes based on pixel values.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 RGB images with 10 classes:
- **Class Labels:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

The dataset is preloaded and ready to use via `keras.datasets.cifar10`.

## Steps Involved

1. **Data Loading & Preprocessing:**
   - Download the CIFAR-10 dataset.
   - Normalize image pixel values to the range `[0, 1]`.
   - One-hot encode target labels for classification.

2. **Model Building:**
   - Build a **Convolutional Neural Network** (CNN) using **Keras Sequential API**.
   - Layers include **Conv2D**, **MaxPooling**, **Flatten**, and **Dense** layers.
   - Activation functions used: **ReLU** (Rectified Linear Unit) and **Softmax** for classification.

3. **Model Training:**
   - Train the model on the training set for 15 epochs using the **categorical crossentropy** loss function and **RMSprop** optimizer.

4. **Evaluation & Metrics:**
   - Evaluate the model using the test set.
   - Generate a **classification report** with precision, recall, and F1-score for each class.

## Model Architecture

The CNN model consists of the following layers:
1. **Conv2D**: Convolutional layer with 12 filters, kernel size `(3, 3)`, and ReLU activation.
2. **MaxPool2D**: Max pooling layer with pool size `(2, 2)`.
3. **Flatten**: Flatten the feature map to a 1D vector.
4. **Dense**: Fully connected layer with 128 units and ReLU activation.
5. **Dense**: Output layer with 10 units (one for each class) and **Softmax** activation.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

## Results

- **Training Accuracy:** 88.35% after 5 epochs.
- **Test Accuracy:** 61.5%
- **Precision, Recall, and F1-score**: Report generated using the `classification_report` from **scikit-learn**.

### Example Precision Report:
```
precision    recall  f1-score   support
0      0.66     0.69     0.68       1000
1      0.75     0.73     0.74       1000
...
accuracy    0.62    10000
macro avg   0.62     0.62     0.62     10000
```

## Results Interpretation

The model demonstrates decent performance, but there's room for improvement. The **confusion matrix** reveals the areas where the model struggles (such as classes with lower precision and recall). Future enhancements could include:
- Hyperparameter tuning
- Advanced augmentation techniques
- Using deeper or more advanced CNN architectures like ResNet or VGG

## Dependencies

- Python 3.x
- **TensorFlow**: For deep learning and CNN implementation
- **Keras**: For building the neural network model
- **NumPy**: For numerical operations
- **Matplotlib**: For visualizations
- **Scikit-learn**: For generating classification reports

To install the necessary dependencies:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

## Conclusion

This project provides a basic introduction to **image classification** using **CNNs** on the CIFAR-10 dataset, showcasing key concepts in **deep learning**, **machine learning**, and **computer vision**. The model achieves respectable accuracy, but there are many opportunities to improve its performance. This can be an excellent base for anyone looking to explore deep learning models in **image recognition**, **object detection**, or **computer vision** tasks.

## Keywords
- **Deep Learning**
- **Convolutional Neural Networks**
- **Computer Vision**
- **Machine Learning**
- **Image Classification**
- **CIFAR-10 Dataset**
- **TensorFlow**
- **Keras**
- **Neural Networks**
- **Python**

