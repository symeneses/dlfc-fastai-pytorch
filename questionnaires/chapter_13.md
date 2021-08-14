# Questionnaire: Convolutional Neural Networks

---
1. What is a "feature"?
---
It's a variable used as input to train a model. It can be a transformation of the raw data to provide a better representation for a given model. Feature engineering is the process of creating new features using different transformations.

---
2. Write out the convolutional kernel matrix for a top edge detector.
---
A convolutional kernel to detect top edges should have higher values to the bottom as there will be higher pixels values in that area.
'''py
top_edge = tensor([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]]).float()
'''
or
'''py
top_edge = tensor([[ 0, 0, 0],
                   [-1,-1,-1],
                   [ 1, 1, 1]]).float()
'''

---
3. Write out the mathematical operation applied by a 3×3 kernel to a single pixel in an image.
---
The kernel will be multiplied by the cells around the given pixel and aggregated (eg: min, max, sum or mean).
'''py
def apply_kernel(row, col, kernel):
    return (im3_t[row-1:row+2,col-1:col+2] * kernel).sum()
'''

---
4. What is the value of a convolutional kernel apply to a 3×3 matrix of zeros?
---
0

---
5. What is "padding"?
---
Padding is adding rows and columns to the input image to be able to calculate values over its edges.

---
6. What is "stride"?
---
It's the distance between the center pixel of the convolution. A stride higher than 1 will decrease the output size.

---
7. Create a nested list comprehension to complete any task that you choose.
---
NA

---
8. What are the shapes of the input and weight parameters to PyTorch's 2D convolution?
---
The [torch.nn.functional.conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html) parameters have the shapes:
- input: (batch_size, in_channels, image_height, image_width)
- weight: (out_channels, in_channels, kernel_height, kernel_width)

---
9.  What is a "channel"?
---
For images channels represent colors. For greyscale images, there is only one channel, but for full-color images, we need one channel for each basic color (R - red, G - green, B - blue).

---
10. What is the relationship between a convolution and a matrix multiplication?
---
A convolution is a special case of a matrix multiplications, where some values are repeated in the rows and there are some zeros to avoid multiplication with unrelated pixels.

---
11. What is a "convolutional neural network"?
---
It's a NN that learns the filters or kernels that are more efficient for a given task.

---
12. What is the benefit of refactoring parts of your neural network definition?
---
It helps to identify and avoid mistakes, improves readability  making clear what is constant and what is variable. 

---
13. What is Flatten? Where does it need to be included in the MNIST CNN? Why?
---
It removes the dimensions with size 1. It's needed in the CNN because the last layer's shape is 64x2x1x1 and we want to have as output a tensor with shape 64x2 where 64 is the batch size and 2 the number of labels.

---
14. What does "NCHW" mean?
---
It indicates the dimensions order in PyTorch convolutional networks.
N: batch size
C: number of channels
H: height
W: width

---
15. Why does the third layer of the MNIST CNN have 7*7*(1168-16) multiplications?
---
First, let's see from where 1168 came from:
$$ n_{params} = ni*nf*ks^2 + nf$$

In the third layer, we have 8 input channels 8 and 16 output channels. The kernel size is 3 for all layers, therefore:

$$ n_{params} = 8*16*3^2 + 16 = 1168$$

Note that the last term corresponds to the bias.

Now, for the total multiplications:

$$ n_{mult} = gride\_size_i*(n_{params} - nf)$$

The gride or activation map size outout of the second layer is 7x7.

$$ n_{mult} = 7*7*(1168 - 16) = 56448$$


---
16.  What is a "receptive field"?
---
It's the area of the input image used as to calculate the activation in each layer.

---
17.  What is the size of the receptive field of an activation after two stride 2 convolutions? Why?
---
It's 7x7. The second convolution will see 3x3 cells in each filter, which corresponds to a receptive area of 7x7 in the input image.

---
18. Run `conv-example.xlsx` yourself and experiment with trace precedents.
---
NA

---
19. Have a look at Jeremy or Sylvain's list of recent Twitter "like"s, and see if you find any interesting resources or ideas there.
---
NA

---
20. How is a color image represented as a tensor?
---
A color image is represented as a 3D tensor. The first dimension represents the color of the pixel and has 3 channels: Red, green and blue.

---
21. How does a convolution work with a color input?
---
It works the same way as a greyscale image, with a kernel of size `channels` x `kernel_size`. 

---
22.  What method can we use to see that data in `DataLoaders`?
---
We can use the `show_batch` method in a transformed `DataLoader` (see [TfmdDL](https://docs.fast.ai/data.core.html#TfmdDL)) to decode samples.

---
23. Why do we double the number of filters after each stride-2 conv?
---
After each stride-2 convolution the activations decrease, but we want to increase the computations in uppers layers to capture more complex features. To achieve that, we double the number of filters.

---
24. Why do we use a larger kernel in the first conv with MNIST (with `simple_cnn`)?
---
As we are increasing the number of filters, if the kernel is small the number of outputs will be close to the number of inputs. To train any model, the outputs have to be smaller than the inputs, using a larger kernel we can achieve that.

---
25. What information does `ActivationStats` save for each layer?
---
This callback records the mean, standard deviation and histogram of the activations. This is helpful to detect problems while training eg. too many activations near zero. 
 
---
26. How can we access a learner's callback after training?
---
The information saved can ba accessed using `learn.activation_stats` and plot with `plot_layer_stats`.

---
27. What are the three statistics plotted by `plot_layer_stats`? What does the x-axis represent?
---
It plots the `mean`, standard deviation: `std` and the percentage of activations near zero: `% near zero` for a given layer. The x-axis is the step number.

---
28. Why are activations near zero problematic?
---
Multiplying by zero is always zero, so the zeros propagate through the network. Which means that most of the network is inactive.

---
29. What are the upsides and downsides of training with a larger batch size?
---
Increasing the batch size helps to stabilize the model, but it will decrease the number of weights updates and increase memory needed to allocate the batch.

---
30.  Why should we avoid using a high learning rate at the start of training?
---
Starting with a high learning rate will make impossible for the model to converge.

---
31. What is 1cycle training?
---
It's a training scheduler or policy where the model is trained starting with a small LR which increases till reaching a maximum value (warmup), followed by a LR decrease (annealing). During the last iterations, the LR is decreased several orders of magnitude to achieve optimal accuracy.

---
32.  What are the benefits of training with a high learning rate?
---
1cycle training helps us to train under relatively higher learning rates, which 1) speeds up the convergence (super-convergence) and 2) reduces the risk of getting stuck in a local minima.

---
33. Why do we want to use a low learning rate at the end of training?
---
Moving at smaller steps to the end of the gradient descent helps to get as closer as possible to the global minima. 

---
34. What is "cyclical momentum"?
---
Momentum uses gradients as acceleration instead of speed which helps gradient descent to converge faster. A `cyclical momentum` in the 1cycle would increase when LR is decreasing and decrease when LR is increasing as the total regularization should remain balanced.

---
35. What callback tracks hyperparameter values during training (along with other information)?
---
This information is saved in `learn.recorders` corresponding to the `Recorder` callback.

---
36. What does one column of pixels in the `color_dim` plot represent?
---
Each column represents the log of the histogram of activations of a batch.

---
37. What does "bad training" look like in `color_dim`? Why?
---
We will see cycles of non-zero (dark blue) and near zero (yellow) activations increasing and decreasing, which means the training is restarting and therefore inefficient.

---
38. What trainable parameters does a batch normalization layer contain?
---
Batch normalization normalizes the activations of the current batch before the activation function of each layer. Along with a right initialization, it helps to reduce gradients problems and reduces the need of regularization.
To tune the `batchnorm` layer, we have two trainable parameters: `gamma` and `beta` so that the activations after normalization are calculated as $y = \gamma*y + \beta$. These parameters adds flexibility to the normalization layers allowing activations to reach optimal values.

---
39. What statistics are used to normalize in batch normalization during training? How about during validation?
---
The mean and standard deviation of each batch are used for normalization while training,for validation the mean of the statistics calculated during training is used.

---
40.  Why do models with batch normalization layers generalize better?
---
They add randomness to the model. Every batch will have different statistics (mean and standard deviation) forcing the model to be more robust to variations in each batch.


## Further Research

---
1. What features other than edge detectors have been used in computer vision (especially before deep learning became popular)?
---
NA

---
2. There are other normalization layers available in PyTorch. Try them out and see what works best. Learn about why other normalization layers have been developed, and how they differ from batch normalization.
---
NA

---
3. Try moving the activation function after the batch normalization layer in conv. Does it make a difference? See what you can find out about what order is recommended, and why.
---
NA
