## Questionnaire: Under the Hood: Training a Digit Classifier

---
1. How is a grayscale image represented on a computer? How about a color image?
---
Grayscale image are represented as numeric vectors where 0 is white, 1 is black and numbers in that range are shades of gray. Color images need 3 channels representing the shade of Red, Green and Blue (RGB) in each pixel.

---
2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
---
The dataset has two folders: `training` and `validation` with sub-folders for every label. This is a standard structure for machine learning datasets.

---
3. Explain how the "pixel similarity" approach to classifying digits works.
---
This simple approach will calculate the average of every pixel and use these averages to define an `ideal` number, which is a representation of the number. Then every image to be classified will be assigned to the closest ideal number.

---
4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
---
A list comprehension is a construct to create a list from another list or iterator.

```py
numbers = [1, 2, 3, 4, 5]
doubled_odd_numbers= [2*i for i in numbers if i%2]
doubled_odd_numbers
```
[2, 6, 10]

---
5. What is a "rank-3 tensor"?
---
It's a tensor with 3 axes or dimensions.

---
6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
---
The tensor rank indicates the number of axes and the shape the size of every axis. The rank is the len of the shape.

---
7. What are RMSE and L1 norm?
---
- **L1**: Mean of absolute errors a.k.a *mean absolute value*
- **RMSE**: Root mean squared error a.k.a *L2* as every number is squared. This loss function gives a higher penalty to bigger errors.

---
8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
---
Using `NumPY` or `PyTorch` functions who are written and optimized in C. 

---
9.  Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
---
```py
import numpy as np

array = 2*np.arange(1, 10).reshape((3, 3))[1:3, 1:3]
array
```
array([[10, 12],
       [16, 18]])

```py
import torch
 
tensor = 2*torch.arange(1, 10).reshape((3, 3))[1:3, 1:3]
tensor
```
tensor([[10, 12],
        [16, 18]])

---
10. What is broadcasting?
---
It's the capability to perform operations between tensors with different rank by expanding the tensor with lower and then .

---
11.  Are metrics generally calculated using the training set, or the validation set? Why?
---  
With the validation set to know if the model is overfitting.

---
12. What is SGD?
---
To learn the weights of a model, we need to optimize a `cost or loss function` (e.g. MSE for regression, Cross Entropy H(p, q)= −Σp(x)log(q(x)) for classification). This optimization is done using `Gradient Descent`, which finds an optimal (normally a minimum) value by measuring the cost function and its gradient every iteration and moving in the descending direction with a `learning rate` size step. If this size is too small, the optimization will be slower and if it's too big, it will lead to an unstable process that may not converge to an optimal value. 
The **Stochastic Gradient Descent (SGD)** does the same but instead of using the whole training set it takes one sample every step, which makes it faster but more unstable.

---
13. Why does SGD use mini-batches?
---
Mini-batch Gradient Descent tries to be fast but also stable:
- If batches are too big, every step will take more time
- Deep learning training is performed in GPUs which are optimized for matrix operations. Training with one sample would be underuse of the GPU.

---
14.  What are the seven steps in SGD for machine learning?
---

1. `Initialize` the weights
2. For each sample, use the weights to `predict` the outputs
3. `Calculate the loss` with these predictions and labels
4. `Calculate the gradient`
5. Move the weights a `step` in the direction the gradient indicates
6. Go back to the step 2, and `repeat` the process
7. `Stop` the iterating process when results are good enough

---
15. How do we initialize the weights in a model?
---
Normally, it's done using random values. There are schemes that help to reduce some training problems. 
In 2010, **Glorot and Bengio** gave light to understand the reasons of these problems: `Variance of the outputs are bigger than the inputs`. They proposed a new initialization (Xavier or Glorot), and in the years after, new schemes have shown better results.

- Glorot: weights are initialized with normal distribution mean 0 and sd=sqr(2/(fan_in + fan_out)) or uniform between -r and r, where r=sqr(6/(fan_in + fan_out))
- He (2015): it uses a truncated normal distribution with mean 0 and sd = sqr(2/(fan_in + fan_out))

For `transfering learning`, the weights are initialized using a **pretrained model**.

---
16.  What is "loss"?
---  
It's a metric to measure the actual model performance and that can be use to adjust the weights.

---
17. Why can't we always use a high learning rate?
---
The weights can't never values that are optimal if the steps are too big. The values may oscillate and never converge.

---
18. What is a "gradient"?
---
The gradient is a function to calculate the rate a function is increasing in every point. If the gradient of a function is non-zero, the direction of the gradient is the direction in which the function increases most quickly if we move from that point. If the gradient is zero, we are at stationary point.

---
1.  Do you need to know how to calculate gradients yourself?
---
To deploy models and be a deep learning practitioner, no. It's recommended to understand their definition, geometric interpretation and applications. 

---
20.  Why can't we use accuracy as a loss function?
---
Accuracy can't give a direction as it's not a continuous function. Its value change only if the predictions change, which is not likely to happen in every step.

---
21. Draw the sigmoid function. What is special about its shape?
---
```py
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-3, 3, 150)
plt.plot(x, torch.sigmoid(x));
```
Sigmoid is used for classification as only gives values between o and 1.

---
22. What is the difference between a loss function and a metric?
---
A metric evaluates the performance of the model. The main difference from loss is its purpose. The metrics is for us humans to compare the model, the loss function is used while training to update weights. For this reason, the loss function must be easy to derivate as that is required for the SGD to determine the direction in which weights should be updated.

---
23. What is the function to calculate new weights using a learning rate?
---
Calculating new weights is a `stepping` of parameters or `optimization step`. The function used for this uses a learning rate to scale the gradients.

$$w -= \nabla w*lr$$

Where
- $w = Weights$
- $lr = Learning\ rate$

In PyTorch:
```py
w -= w.grad*lr
```

---
24.   What does the `DataLoader` class do?
---
It's a fastai class which loads the data to be used for training as `train` and `valid`. It takes as input a collection, can shuffle the data, and outputs an iterator over batches.

---
25.   Write pseudocode showing the basic steps taken in each epoch for SGD.
---

- Initialize vectors of weights *w* and learning rate *lr*.
- Repeat until reaching an approximate minimum value:
    - Randomly shuffle samples in the training set.
    - For i = 1 , 2 , . . . , n , do:
      - $loss_i(w) = loss(prediction_i, y_i)$
      - $gradients_i = \nabla loss_i(w)$
      - $w := w − lr*gradients_i$

---
26.  Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
---
The output is a list of tuples, which is the format used for a `Dataset` in PyTorch.

```py
def tupler(iterator1, iterator2): return list(zip(iterator1, iterator2))

numbers_list = [1,2,3,4]
alphabet_string = 'abcd'

tupler(numbers_list, alphabet_string)
```
[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]

---
27. What does `view` do in PyTorch?
---
It creates a tensor `View` that can be reshaped or sliced without copying the data of the base tensor.

```py
import torch

t_base = torch.rand(4, 3)
t_view  = t_base.view(2, 6)
print(t_base.shape, t_view.shape)
t_view[1, 3] = 0.5
print(t_base[3, 0] == t_view[1, 3])
t_base[3, 0] 
```

---
1.  What are the "bias" parameters in a neural network? Why do we need them?
---
These are the parameters that influence the output no matter the values in the input. When the values are zero, the output will be equal to the bias, which is necessary to give flexibility to the layer.

---
29. What does the @ operator do in Python?
---
It represents matrix multiplication.

---
30. What does the backward method do?
---
The backward method calculates the gradients of the loss function.

---
31. Why do we have to zero the gradients?
---
In PyTorch the gradients will accumulate unless they are set to zero or None again.

---
32. What information do we have to pass to `Learner`?
---
The [`Learner`](https://docs.fast.ai/learner.html#Learner) needs:
- A `DataLoader`
- A PyTorch `model`
- A fastai [`Optimizer`](https://docs.fast.ai/optimizer.html#Optimizer) or a PyTorch function wrapped using [`OptimWrapper`](https://docs.fast.ai/optimizer#OptimWrapper). `Adam` is the default.
- A loss function. If it is wrapped using [`BaseLoss`](https://docs.fast.ai/losses.html#BaseLoss), The [`Learner.predict`](https://docs.fast.ai/learner.html#Learner.predict) method can be used.

---
33.  Show Python or pseudocode for the basic steps of a training loop.
---
```py
def train_epoch(dl, model, opt):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
```

---
34. What is "ReLU"? Draw a plot of it for values from -2 to +2.
---

```py
import torch
from matplotlib import pyplot as plt

x = torch.linspace(-2., 2., 50)
plt.plot(x, torch.relu(x));
```
The Rectifier Linear Unit (ReLU) is a function that replaces negatives values with zeros. It's used in Deep Learning to add nonlinear layers, which are important to achieve complex functions.

---
35.  What is an "activation function"?
---
It's a nonlinearity that is generally applied to the output of a linear function before sending values to the next layer.

---
36. What's the difference between `F.relu` and `nn.ReLU`?
---
`F.relu` is the ReLU PyTorch function and `nn.ReLU` is a *module* that can be added to a PyTorch `model`. A PyTorch `module` is a class to define models as a tree structure, therefore, the model itself is also a module.

---
37.  The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
---
Deep Learning use more nonlinear layers as it is easier and faster to train smaller layers. Stacking layers allow us to create complex functions by adding more nonlinearities. The first layers will generate simple features based on the input and subsequent layers can create more insightful features using these as an input.
It's also important to keep in mind that `Universal approximation theorem` holds true only for continuous functions. 

## Further Research:

---
1. Create your own implementation of Learner from scratch, based on the training loop shown in this chapter.
---
NA

---
2. Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.
---
NA
