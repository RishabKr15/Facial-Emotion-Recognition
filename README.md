# Facial-Emotion-Recognition
This project focuses on detecting seven distinct facial emotions — Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral — using deep learning models trained on the FER-2013 dataset, which consists of 35,887 labeled grayscale images .

#🚀 Project Highlights
## Models Used:

Custom CNN: Tailored for FER-2013 to efficiently learn hierarchical features.

VGG16 (with fine-tuning): Leveraged pre-trained ImageNet weights and added custom classification head.

ResNet50 (Transfer Learning): Achieved high validation accuracy by freezing base layers and tuning dense layers.

## 🎯 Performance:

Achieved up to 72% validation accuracy by experimenting with various architectures, fine-tuning techniques, and layer configurations.

🔄 Data Preprocessing & Augmentation:

Performed real-time data augmentation using Keras ImageDataGenerator.

## Techniques included:

Rescaling

Horizontal flipping

Random rotation

This improved model generalization and reduced overfitting by 20%+.

## Training Optimization:

Integrated EarlyStopping to prevent unnecessary training once the model stopped improving.

Used ModelCheckpoint to automatically save the best-performing model.

Together, these reduced training time by 15%, increased efficiency, and ensured model rollback for optimal performance.

## Tools & Libraries
Python, TensorFlow, Keras

NumPy, Matplotlib, Seaborn

Google Colab / Jupyter Notebook


## Dataset Link : https://www.kaggle.com/datasets/msambare/fer2013
