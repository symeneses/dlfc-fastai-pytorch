## Questionnaire: Image Classification

---
1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
---
Resize to larger dimensions is a technique to avoid empty spaces due to augmentation operations (rotations or cropping) while training.  The augmentations should be done in the GPU as batch operations to do it faster including the resize to the target size. 

---
2. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
---
NA

---
3. What are the two ways in which data is most commonly provided, for most deep learning datasets?
---
- Files as text documents or images where commonly their paths give some details about them
- Tabular data where each row is an item and its details are in the table columns

---
4. Look up the documentation for L and try using a few of the new methods that it adds.
---
The L class is a python list with additional methods inspired by NumPy
https://fastcore.fast.ai/foundation#L

```bash
pip install fastcore
```

```py
from fastcore.foundation import L
import operator

super_list = L(range(1, 11))
super_list
```
(#10) [1,2,3,4,5,6,7,8,9,10]

```py
super_list.map_dict(lambda x:x**2)
```
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81, 10: 100}

```py
super_list = L(super_list, [x**2 for x in range(1, 11)])
super_list.zip()
```
(#10) [(1, 1),(2, 4),(3, 9),(4, 16),(5, 25),(6, 36),(7, 49),(8, 64),(9, 81),(10, 100)]

```py
super_list.zip().filter(lambda x: x[0]%2)
```
(#5) [(1, 1),(3, 9),(5, 25),(7, 49),(9, 81)]

```py
super_list.map_zip(operator.mul)
```
(#10) [1,8,27,64,125,216,343,512,729,1000]

---
5. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.
---

https://docs.python.org/3/library/pathlib.html#module-pathlib
```py
# Code executed on Google colab
from pathlib import Path

print(Path.cwd())
p = Path('/content/sample_data')
print(p.is_dir())
Path('/content/sample_data/test').mkdir()
[(f, f.is_file()) for f in p.iterdir()]
```
/content
True
[(PosixPath('/content/sample_data/README.md'), True),
 (PosixPath('/content/sample_data/anscombe.json'), True),
 (PosixPath('/content/sample_data/.ipynb_checkpoints'), False),
 (PosixPath('/content/sample_data/test'), False),
 (PosixPath('/content/sample_data/california_housing_test.csv'), True),
 (PosixPath('/content/sample_data/california_housing_train.csv'), True),
 (PosixPath('/content/sample_data/mnist_test.csv'), True),
 (PosixPath('/content/sample_data/mnist_train_small.csv'), True)]

---
6. Give two examples of ways that image transformations can degrade the quality of the data.
---
- Transformations can lead to empty zones from which the model won't learn anything.
- While resizing or rotating, values in every pixel must be interpolated. This causes lost of information.

---
7. What method does fastai provide to view the data in a DataLoaders?
---
The `one_batch` method allows us to see the activations of the model and labels.  

---
8. What method does fastai provide to help you debug a DataBlock?
---
The `DataBlock` class has the method `summary` which shows in detail every step performed while creating a batch. 

---
9.  Should you hold off on training a model until you have thoroughly cleaned your data?
---
No, training a model is an iterative process. It's better to get results earlier and improve every step gradually until reaching the expected performance. 

---
10. What are the two pieces that are combined into cross-entropy loss in PyTorch?
---
The [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines:
- [`log_softmax`](https://pytorch.org/docs/stable/nn.functional.html?highlight=log_softmax#torch.nn.functional.log_softmax): Applies a softmax followed by a logarithm $log(softmax(x))$.
- [`nll_loss`](https://pytorch.org/docs/stable/nn.functional.html#nll-loss): The negative log likelihood loss.

---
11. What are the two properties of activations that softmax ensures? Why is this important?
---
- Output values are always between 0 and 1
- The sum of the values in a layer are 1
These properties are useful to interpret the results as probabilities.

---
12. When might you want your activations to not have these two properties?
---
If we want to give weight to small changes, we need to get values between minus infinite and infinite. For this reason, the log function is used after the softmax.

---
13.  Calculate the exp and softmax columns of Figure 5-3 yourself (i.e, in a spread-sheet, with a calculator, or in a notebook)
---
```py
from torch import exp, nn, tensor, zeros

matrix = zeros([3, 3])
matrix[:, 0] = tensor([0.02, -2.49, 1.25])
matrix[:, 1]  = exp(matrix[:, 0])
matrix[:, 2] = nn.Softmax(dim=1)(matrix[:, 1:2].T)
matrix
```
tensor([[ 0.0200,  1.0202,  0.0757],
        [-2.4900,  0.0829,  0.0296],
        [ 1.2500,  3.4903,  0.8947]])

---
14. Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?
---
The function `torch.where` validates that a given condition is true or false, for many categories we have a value for every category, in this case, it's easier to use the indexes to select the right values.

---
15.  What is the value of log(-2)? Why?
---
The log function is not defined for negative numbers. In python, it will generate an exception `ValueError: math domain error`.

---
16. What are two good rules of thumb for picking a learning rate from the learning rate finder?
---
- To select one order of magnitude less from the point the loss reaches its minimum value
- Last point where the loss was decreasing

---
17. What two steps does the `fine_tune` method do?
---
- Train the new layers for one epoch while keeping the pre-trained ones frozen
- Unfreeze all layers and train the model for the given epochs

---
18.  In Jupyter Notebook, how do you get the source code for a method or function?
---
```py
function??
```

---
19. What are discriminative learning rates?
---
It's a the use of different learning rates according to the layer, earlier layers which are pretrained should learn slower than posterior layers which need the features created in the previous layers. In fastai, we can give a python slice to define the learning rate in the parameter `lr_max`.

---
20. How is a Python slice object interpreted when passed as a learning rate to fastai?
---
The first value will be the learning rate for the earliest layer, equidistant values between the slice range will be applied to the middle layers and the second value will be used for the final layer.

---
21. Why is early stopping a poor choice when using `1cycle` training?
---
1cycle learning decreases the learning rate at the end of one cycle and it's at that training stage where the model can reach optimal performance. If we use `early stopping`, the model may not overfit but with a 1cycle policy, we can get better results if the model is trained from scratch for the number of epochs where the best results were previously observed.

---
22.  What is the difference between `resnet50` and `resnet101`?
---
The number specifies the *capacity* of the model in terms of layers and therefore parameters.

---
23. What does `to_fp16` do?
---
It converts the model tensors to half-precision floating point or fp16. This speeds up training by 2-3x and reduces GPU memory consumption.
  
## Further Research

---
1. Find the paper by Leslie Smith that introduced the learning rate finder, and read it.
---
Leslie Smith introduced the learning rate finder in 2015, and since then he has written these related papers.
- 2015 [No more pesky learning rate guessing games](https://arxiv.org/pdf/1506.01186v2.pdf)
- 2017 [Cyclical learning rates for training neural networks](https://arxiv.org/pdf/1506.01186.pdf)


>It’s widely accepted that the learning rate is the most important hyperparameter in deep learning. Decreasing learning rates is a common technique to reach optimal values, the paper proposes cyclical training rates (CLR) which have cycles where the learning rate linearly (`triangular policy`) or exponentially (`exp_range policy`) increases and then decreases within a range of values. Experimental results indicate that the stepsize (iterations to reach the maximum value) should be set to 2-8 times the iterations per epoch. After 3 cycles or 6*stepsize iterations, the training is almost done and the learning rates should drop till reach optimal values. 
To find the LR range, the author suggested training the model for 4-8 epochs starting with a low LR and taking as reference the values where the accuracy starts to fall.


- 2018 [A disciplined approach to neural network hyper-parameters: Part 1--learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820.pdf)

> The author suggests training a DL model for a few epochs to gain useful insights about the data and architecture which facilitates setting suitable hyperparameters instead of performing resource intensive hyperparameter tuning. Observing the test loss, we can see if the model is underfitting, in this case, the test loss will decrease continuously. On the other hand, if it increases after dropping, the model is overfitting. 
A new policy `1cycle` is introduced, under this policy the model is trained for 1 cycle with less iterations than its predecessor `linear_policy` and instead, during the last iterations, the LR is decreased several orders of magnitude. In the paper, it is demonstrated that following this policy training can reach optimal accuracy.
While this policy shows to be universal, the amount of regularization introduced through large learning rates, small batch sizes, weight decay, and dropout should be balanced for every problem. This means, for example, if we are using large learning rates, then weight decay should be kept within some limits and larger batch sizes are recommended. 
For weight decay, the author proposes an alternate cycle (decreasing when LR is increasing and vice versa) while the weight decay should remain constant. Grid search is recommended to find a suitable weight decay with values between 10^- 6 dnd 10^-4 for complex datasets and 10^-4 and 10^-2 for smaller ones. 


---
2. See if you can improve the accuracy of the classifier in this chapter. What's the best accuracy you can achieve? Look on the forums and the book's website to see what other students have achieved with this dataset, and how they did it.
---
NA
