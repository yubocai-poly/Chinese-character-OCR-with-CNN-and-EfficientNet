# OCR for Chinese Characters with CNN and EfficientNet

> This is the final project of CSE204 Machine Learning course in Ecole Polytechnique. We use CNN and EfficientNet model to do OCR for singular handwriting Chinese Characters. The project is mainly based on Tensorflow 2.0 API.

## Contributors

- [Junyuan Wang](https://github.com/frank2002)
- [Yubo Cai](https://github.com/yubocai-poly)

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

**Note:** If you are using
