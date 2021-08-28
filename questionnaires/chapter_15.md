# Questionnaire: Application Architectures Deep Dive

--
1. What is the "head" of a neural net?
---
It's the last few layers of the network, which are trained when we apply transfer learning to adapt the model to a specific task. The head can be customized according to the task complexity and number of categories.

---
2. What is the "body" of a neural net?
---
The body entails the layers needed to kept information about a domain making it useful for transfer learning. The body of a pre-trained model is frozen as it has already learned on a larger dataset.

---
3. What is "cutting" a neural net? Why do we need to do this for transfer learning?
---
It's splitting a pre-trained model into head and body. As each architecture can have a different cut point, this is configurable. The body will be used for transfer learning plus a customized head created for a specific task.

---
4. What is `model_meta`? Try printing it to see what's inside.
---
```python
from fastai.vision.all import model_meta

model_meta
```

It's a dictionary that contains information about multiple torchvision models, with the model functions as keys. The values are dictionaries containing the cutting point `cut`, the split function `split` and the statistics (from the weights) `stats` for each model.

---
5. Read the source code for `create_head` and make sure you understand what each line does.
---
NA

---
6. Look at the output of `create_head` and make sure you understand why each layer is there, and how the `create_head` source created it.
---
NA

---
7. Figure out how to change the dropout, layer size, and number of layers created by `cnn_learner`, and see if you can find values that result in better accuracy from the pet recognizer.
---
NA

---
8. What does `AdaptiveConcatPool2d` do?
---
It's a layer that concatenates an average pooling layer with a max pooling layer. It has been shown experimentally to improve the model performance.

---
9.  What is "nearest neighbor interpolation"? How can it be used to upsample convolutional activations?
---
It's an upsampling method that takes a value for each pixel and applies in a nxm grid to get a output with a larger size as shown in the code snippet.

```python
import torch
from torch import nn

input = torch.randn(1, 3, 7, 7)
output = nn.functional.interpolate(input, scale_factor=(2, 2), mode='nearest')
output.shape
```
`torch.Size([1, 3, 14, 14])`

```python
input[0][0][0]
```
`tensor([-0.0099,  1.4801,  0.1446,  0.2114, -0.9325,  0.5107, -0.1421])`

```python
output[0][0][0:3, 0:5]
```
`tensor([[-0.0099, -0.0099,  1.4801,  1.4801,  0.1446],
        [-0.0099, -0.0099,  1.4801,  1.4801,  0.1446],
        [ 0.4405,  0.4405,  1.4716,  1.4716, -0.8124]])`

We can see that each value appears 4 times in the output. The output size is (7*2)x(7*2)=14x14.

---
10.  What is a "transposed convolution"? What is another name for it?
---
In an *transposed convolution* or *stride half convolution* a zero padding is inserted between every pixel before doing the convolution. This increases the output input size.

---
11. Create a conv layer with `transpose=True` and apply it to an image. Check the output shape.
---
```python
from fastai.vision.all import ConvLayer, image2tensor, Image, download_images, Path, torch

download_images(Path("/content"), urls=["https://github.com/fastai/course20/raw/master/fastbook/images/grizzly.jpg"], preserve_filename=True)
im = image2tensor(Image.open(Path("/content/grizzly.jpg")))
input = torch.unsqueeze(im, 0)/255
input.shape
```
`torch.Size([1, 3, 1000, 846])`

```python
conv_trans = ConvLayer(3, 6, transpose=True)
output = conv_trans(input)
output.shape
```
`torch.Size([1, 6, 1002, 848])`

The output size is (100+2)x(846+2)=1002x848.

---
12. Draw the U-Net architecture.
---
![unet-arch](https://raw.githubusercontent.com/HansBambel/SmaAt-UNet/master/SmaAt-UNet.png "U-NET Architecture")

*Source: https://github.com/HansBambel/SmaAt-UNet*

---
13. What is "BPTT for Text Classification" (BPT3C)?
---
Backpropagation Through Time (BPTT) is a training technique to calculate the gradients for RNN, which update the weights based on accumulated errors across each timestep. BPTT3C is a variant which divide a text in fixed-length batches, keeping the activations across batches. At then end, average and max pooling are calculated with the saved activations.

---
14. How do we handle different length sequences in BPT3C?
---
Batches are created randomly with sequences of similar length to avoid useless calculations. The sequences are then padded (with `xxpad`) to get fixed-length batches.

---
15. Try to run each line of `TabularModel.forward` separately, one line per cell, in a notebook, and look at the input and output shapes at each step.
---
NA

---
16. How is `self.layers` defined in `TabularModel`?
---
It's a sequential model (`nn.Sequential`) that contains `LinBnDrop` layers. A `LinBnDrop` layers groups `BatchNorm1d`, `Dropout` and `Linear` layers.

---
17.  What are the five steps for preventing over-fitting?
---
1. More data: add labels, tasks, etc.
2. Data augmentation: create synthetic data.
3. Generalizable architectures: add batch normalization.
4. Regularization: add dropout.
5. Reduce architecture complexity: use a smaller model.

---
18. Why don't we reduce architecture complexity before trying other approaches to preventing overfitting?
---
A small model is not able to learn complex relationships, therefore, it's recommended to take a data-driven approach first. Then, try regularization as it can lead to more flexible models.


## Further Research

--
1. Write your own custom head and try training the pet recognizer with it. See if you can get a better result than fastai's default.
---
NA

---
3. Try switching between `AdaptiveConcatPool2d` and `AdaptiveAvgPool2d` in a CNN head and see what difference it makes.
---
NA

---
4. Write your own custom splitter to create a separate parameter group for every `ResNet` block, and a separate group for the stem. Try training with it, and see if it improves the pet recognizer.
---
NA

---
5. Read the online chapter about generative image models, and create your own colorizer, super-resolution model, or style transfer model.
---
NA

---
6. Create a custom head using nearest neighbor interpolation and use it to do segmentation on CamVid.
---
NA