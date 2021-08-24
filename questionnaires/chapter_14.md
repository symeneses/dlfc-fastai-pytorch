# Questionnaire: ResNets

---
1. How did we get to a single vector of activations in the CNNs used for MNIST in previous chapters? Why isn't that suitable for Imagenette?
---
For MNIST, we used a stack of stride-2 convolutions till reaching a grid size of 1. This approach is not scalable as 1) the number of layers would depend on the input size and 2) the model would only work for a fixed input size.

---
2. What do we do for Imagenette instead?
---
A fully convolutional network, which has a stack of convolutional layers followed by a pooling layer (usually average pooling). The pooling layer takes the average (or maximum) of the activations of grid, therefore, it can be calculated irrespective of the grid size.

---
3. What is "adaptive pooling"?
---
It's a pooling layer that allows to set any output size. The layer will calculate the stride and kernel size so that no matter the input size, we can always the desired output size.
Adaptative pooling is available in PyTorch with the layers: [AdaptiveMaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html#torch.nn.AdaptiveMaxPool2d) and [AdaptiveAvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html#torch.nn.AdaptiveAvgPool2d).

---
4. What is "average pooling"?
---
It's a pooling layer that takes the average of the activations. In the snippet, we use adaptive average pooling to transform the input (8x9) to (5x7).

```python
from torch import nn

m = nn.AdaptiveAvgPool2d((5,7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
output.shape
```
torch.Size([1, 64, 5, 7])

---
5. Why do we need `Flatten` after an adaptive average pooling layer?
---
To remove the dimension of size 1 before applying the linear layer.

---
6. What is a "skip connection"?
---
A skip connection gives an alternate path to the gradient descent by feeding the output of one layer to next layers, that means it could be to more than one and not to a consecutive one. This input to next layers can be represented as $x + conv2(conv1(x))$, where x is *skipping* the convolutional layers.

---
7. Why do skip connections allow us to train deeper models?
---
In theory, a neural network can learn any function, but for practical purposes, it has to learn with the available data and resources. Skip connections simplify the function the model has to learn as it's learning the residuals.

---
8. What does Figure 14-1 show? How did that lead to the idea of skip connections?
---
The figure shows that larger models have higher errors even in the training set, which means the error is not due to overfitting but to the training process. The authors of the ResNet paper noticed that a larger model can be created using a smaller one and adding layers that do nothing, and then train the new layers to improve the model. This idea led them to create skip connections, which instead of learning the output are learning to reduce the error.

---
9.  What is "identity mapping"?
---
It's a function that doesn't modify the input, that is $f(x) = x$.

---
10. What is the basic equation for a ResNet block (ignoring batchnorm and ReLU layers)?
---
$$
Convolution\ Part + Skip\ Connection = F(x) + x = conv2(conv1(x)) + x
$$

---
11. What do ResNets have to do with residuals?
---
As the ResNet block is passing the input to the output directly, the layer has to learn only the difference between the input and the output a.k.a as residuals.

---
12. How do we deal with the skip connection when there is a stride-2 convolution? How about when the number of filters changes?
---
As the skip connection is added to the output of the convolutional layer, it has to match its output size, therefore, 1) if the stride is different to 1, an average pooling layers (`nn.AvgPool2d`) is added to the rest block, and 2) if the number of filters changes (ni!=nf), a 1x1 convolutional layer (`ConvLayer`) is added to the rest block.

---
13. How can we express a 1×1 convolution in terms of a vector dot product?
---
A 1x1 convolution is equivalent to a multilayer perceptron that acts as a cross-channel parametric pooling layer.

---
14. Create a 1x1 convolution with `F.conv2d` or `nn.Conv2d` and apply it to an image. What happens to the shape of the image?
---
```python
from fastai.vision.all import image2tensor, Image, download_images, Path, nn, torch, TensorImage

download_images(Path("/content"), urls=["https://github.com/fastai/course20/raw/master/fastbook/images/grizzly.jpg"], preserve_filename=True)
im = image2tensor(Image.open(Path("/content/grizzly.jpg")))
input.shape
```
torch.Size([1, 3, 1000, 846])

```python
input = torch.unsqueeze(im, 0)/255
conv1x1 = nn.Conv2d(3, 5, 1)
output = conv1x1(input)
output.shape
```
torch.Size([1, 5, 1000, 846])

After applying the 1x1 convolution, the output size is the same as the input size.

---
15. What does the noop function return?
---
`noop` stands for no operation and represents a function that does nothing, and therefore, returns its input.

---
16. Explain what is shown in Figure 14-3.
---
The figure shows how the `ResNet` makes the gradient descent smoother, speeding up the training and helping to skip local minima. 

---
17. When is top-5 accuracy a better metric than top-1 accuracy?
---
When each sample can have multiple correct labels or there is a high chance or mislabels. Top-5 accuracy will measure that the label is within the top 5 predictions, which is a better indication of model performance in these cases. 

---
18.  What is the "stem" of a CNN?
---
It's the first convolutional layers. 

---
19. Why do we use plain convolutions in the CNN stem, instead of ResNet blocks?
---
In the case of a ResNet, the stem can have normal convolutional layers which are simpler (in comparison with a ResNet block which has three convolutions and a pooling layer) as the first layers are computationally more intensive. 

---
20. How does a bottleneck block differ from a plain ResNet block?
---
A bottleneck block has a three layers: 1) convolutional layer with kernel size 1x1, 2) convolutional layer with kernel size 3x3, and 3) a convolutional layer with kernel size 1x1. This configuration reduces the parameters and increases the filters, allowing deeper models.

---
21. Why is a bottleneck block faster?
---
As the number of parameters decrease, we have less matrix multiplications to be executed. 

---
22. How do fully convolutional nets (and nets with adaptive pooling in general) allow for progressive resizing?
---
Progressive resizing requires models that are flexible to the input size as the idea is to start learning with small images and then increase the size of the input. Fully convolutional nets can be used with images of any input size, which is precisely the need in this case.


## Further Research

---
1. Try creating a fully convolutional net with adaptive average pooling for MNIST (note that you'll need fewer stride-2 layers). How does it compare to a network without such a pooling layer?
---
NA

---
2. In Chapter 17, we introduce *Einstein summation notations*. Skip ahead to see how this works, and then write an implementation of the 1x1 convolution operation using `torch.einsum`. Compare it to the same operation using `torch.conv2d`.
---
NA

---
3. Write a "top-5 accuracy" function using plain PyTorch or plain Python.
---
NA

---
4. Train a model on Imagenette for more epochs, with and without label smoothing. Take a look at the Imagenette leaderboards and see how close you can get to the best results shown. Read the linked pages describing the leading approaches.
---
NA
