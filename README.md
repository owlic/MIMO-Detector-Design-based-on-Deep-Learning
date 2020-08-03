# MIMO Detector Design based on Deep Learning

## Introduction
This repositiry is about using deep neural networks to build MIMO(multiple-input and multiple-output) detectors.

The architecture of DetNet is referenced by https://arxiv.org/abs/1706.01151.



## How to use
**Step1: adjust the hyperparameters in `parameters.py` as needed.**

**Step2: use `generate_data.py` to synthesize the validation dataset.**

**Step3: let's start training.**



## Environment
Main package|Version
---|---
Python|3.7.4
Matplotlib|3.1.1
Numpy|1.16.5
TensorFlow|1.14*

*_If you wanna run on TensorFlow 2, add `tf.compat.v1.disable_eager_execution()` in `main.py`._
