# OCR for Chinese Characters with CNN and EfficientNet

> This is the final project of CSE204 Machine Learning course in Ecole Polytechnique. We use CNN and EfficientNet model to do OCR for singular handwriting Chinese Characters. The project is mainly based on Tensorflow 2.0 API.

## Contributors

- [Junyuan Wang](https://github.com/frank2002)
- [Yubo Cai](https://github.com/yubocai-poly)

## Table of Contents

- [Introduction](#1-introduction)
- [Packages Prerequisites](#2-packages-prerequisites)
- [Data Preparation](#3-dataset-preparation)
- [Model Construction](#4-model-construction)
- [Model Training](#5-model-training)
- [Results of the Models](#6-results-of-the-model)
- [Download Link for Datasets and Model Checkpoints](#7-download-link-for-datasets-and-model-checkpoints)
- [References](#8-bibliography)

## Some Notification

For all the dataset and trainning checkpoints, you can got to [Part 7](#7-download-link-for-datasets-and-model-checkpoints). You need to place the dataset in the `./dataset` folder and the checkpoints in the main folder of this project.

If you are using **Mac OS** and have problem label the Chinese characters with `Matplotlib`, this [document](MacLabel_Problem.md) may help you.

## 1. Introduction

When we learn CNNs in machine learning, we used the MNIST dataset for image recognition. This data set contains 26 letters of the alphabet as well as numbers. Therefore we would like to try OCR recognition for Chinese handwritten characters. However, the difficulty of this is very high compared to MNIST.

|  | Chinese Characters | MNIST (English Alphabet) |
| --- | --- | --- |
| Number of Characters | 10k+ | 26 |
| Structure | More Complex | Relatively simple |
| Diversity of writing styles | A large number of different writing styles as well as continuous strokes | More fixed |

The main reasons why Chinese handwritten OCR is more difficult than MNIST are the following:

1. There are **more than 10k** Chinese Characters which is much more than the 26 letters in MNIST. Therefore, it's quite difficult to label the data and build a complex model to recognize all the characters with high accuracy.

2. Chinese characters are more complex in shape. Compared the MNIST dataset, Chinese characters have a higher number of strokes, more complex shapes, and more variants and fonts exist, thus requiring more advanced image processing and feature extraction techniques for recognition.

3. There are numerous variations and styles of Chinese writing, such as Cursive (草书), Semi-cursive script (行书) and Regular script (楷书). [[More information about Chinese script styles]](https://en.wikipedia.org/wiki/Chinese_script_styles).
 
<div align=center>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/7/73/Cur_eg.svg" width="180">
    <figcaption>(草书)</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/55/Semi-Cur_Eg.svg" width="180">
    <figcaption>(行书)</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Kaishu.png" width="180">
    <figcaption>(楷书)</figcaption>
  </figure>
</div>

## 2. Packages Prerequisites

Here we provide the requirements.txt file for the packages prerequisites. You can install all the packages by running the following command:

```bash
pip install -r requirements.txt
```

**Note:** If you are using **Apple Silicon M1/M2**, the installation of `tensorflow` may fail. You can find the solution in [this webpage](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706).

## 3. Dataset Preparation

Here we use the data from [CASIA's open source Chinese handwriting dataset](http://www.nlpr.ia.ac.cn/databases/handwriting/GTLC.html). In our project, we only use the **Offline CASIA-HWDB 1.0-1.2 Database**. All data in the version 1.0-1.2 are single-word and 2.0-2.2 is sentence-level which require **CRNN**. This dataset contains 7185 Chinese characters and 171 English letters, numbers, punctuation, etc. More detailed information in this [link](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html).

You can operate the following command to download the dataset:

```bash
cd local_address_to_this_project/dataset
chmod u+x get_hwdb_1.0_1.1.sh
get_hwdb_1.0_1.1.sh
```

After downloading the dataset and unzip it, you can find a **train** folder and a **test** folder. However, the format of the dataset is `gnt` with 32 characters and labels which is not convenient for us for trainning. Therefore, we provide a script to convert the dataset to `png` format with labels. We can run the following command to convert the dataset into `tfrecord` format:

```bash
cd local_address_to_this_project/dataset
python3 convert_to_tfrecord.py HWDB1.1tst_gnt
python3 convert_to_tfrecord.py HWDB1.1trn_gnt
```

We also provide the converted dataset in the following link. You can directly use it for training.
- [test.tfrecord](https://drive.google.com/file/d/1knT-6pgkTKmvAp-fivCMUtOU9rRG_X-P/view?usp=sharing)
- [train.tfrecord](https://drive.google.com/file/d/1BhisIm3ebKTLasUx-VNGtIGXYEFJjtlc/view?usp=sharing)

## 4. Model Construction

We use Three models for this project. The first one is a **simple CNN** model, second one is a more **complex CNN** model, and the third one is a **pre-trained EfficientNetB0** model. 

Here we just simply use the `tf.keras` API to build the model to build the CNN model. For the **simple CNN** model, we just add 2 convolutional layer and 2 maxpooling layer. For the **pre-trained EfficientNetB0** model, we use the API from tensorflow as follows:

```python
def effcientnetB0_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # EfficientNetB0 expects 3 channels
    base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model
```

## 5. Model Training

## 6. Results of the model

### 6.1 Simple CNN Model

In fact we were surprised that this simple CNN model had some recognition rate, considering the complex structure of Chinese characters and the large number of writing styles in our dataset. The accuracy of the simple CNN model is around 40% for the `testing dataset` and 97% for `training dataset` after about 60 epoches which is not bad for a simple model.

<div align=center>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_cnn_accuracy.png" width="480">
  </figure>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_cnn_loss.png" width="480">
  </figure>
</div>

We test some random images from the testing dataset and the results are as follows:

<div align=center>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_simple_cnn.png" width="700">
  </figure>
</div>

We can see that there are 4 correct prediction among the 9 images. However, we are not satisfied with the results. Therefore, we try to use a more complex CNN model to improve the accuracy.

### 6.2 Complex CNN Model

Interestingly, after our training and testing, we found that this complex model did not converge and the accuracy was still very low after almost 200 epoches of training. Our analysis suggests the following reasons:

### 6.3 Pre-trained EfficientNetB0 Model

We use the pre-trained EfficientNetB0 model to train our dataset. The accuracy of the EfficientNetB0 model is around 87% for the `testing dataset` and 96% for `training dataset` after about 60 epoches which in a huge improvement compared to the simple CNN model.

<div align=center>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_eff_accuracy.png" width="480">
  </figure>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_eff_loss.png" width="480">
  </figure>
</div>

We test some random images from the testing dataset and the results are as follows:

<div align=center>
  <figure style="display: inline-block; margin: 0 20px;">
    <img src="Graphs/output_EfficientNet.png" width="700">
  </figure>
</div>

We can see that there are 7 correct prediction among the 9 images. We think this is a very good result, considering that we are analysing images of such a complex structure that many words are difficult to distinguish with our observations.

## 7. Download Link for Datasets and model checkpoints

>  **Directly download link for all Datasets and training checkpoints can be founded below**

The link below can be used to download processed data and trained model checkpoints file. The datasets have been already converted to `tfrecord`. The checkpoints provided have a validation accuracy around 40% for simple CNN model and 87% for EfficientNetB0 model.
- [Dataset & Checkpoints](https://frrl.xyz/dataset)
- [Checkpoints only](https://frrl.xyz/ckpt)

### Usage
1. Unzip the files after downloading. 
2. Merge the folder in the zip file into the project root directly.

## 8. Bibliography
