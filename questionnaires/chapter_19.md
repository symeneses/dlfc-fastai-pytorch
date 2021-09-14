# Questionnaire: A fastai Learner from Scratch


---
1. What is glob?
---
[Glob patterns](https://en.wikipedia.org/wiki/Glob_(programming)) are used to filter a set of files with wildcard characters (*, ?). Python implements glob with the [glob module](https://docs.python.org/3/library/glob.html).

---
2. How do you open an image with the Python imaging library?
---
```py
from PIL import Image

img = Image.open('file_path')
```

---
3. What does `L.map` do?
---
It applies a function to every item in the list using the given arguments.

---
4. What does `Self` do?
---
[Self](https://fastcore.fast.ai/basics.html#Self-(with-an-uppercase-S)) is a class from `fastcore` to create lambda functions, so that, these two lines are equivalent.

- Using lamdba:
```py
lbls = files.map(lambda x: x.parent.name).unique(); lbls
```

- Using `Self`:
```py
lbls = files.map(Self.parent.name()).unique(); lbls
```

---
5. What is `L.val2idx`?
---
It creates a dictionary with the elements in the list as keys and a sequential integer as values. This is useful to create vocabularies.

---
6. What methods do you need to implement to create your own Dataset?
---
Class with the `__len__` (returns the items in the dataser) and `__getitem__` (returns a sample corresponding to a given index) methods implemented will create objects that can be a `Dataset` in PyTorch.

---
7. Why do we call convert when we open an image from Imagenette?
---
To be sure that all images are in the same `mode`.

---
8. What does `~` do? How is it useful for splitting training and validation sets?
---
It performs a bitwise NOT or complement (ex: 101 > 010). In numpy, it's used as a negation (`~np.False_ == np.True_`), which makes this operator useful to split data.

---
9.  Does ~ work with the L or Tensor classes? What about NumPy arrays, Python lists, or pandas DataFrames?
---
L: ✅
```python
from fastai.basics import L

~L([1, 2, 3, False, True])
```
`(#5) [False,False,False,True,False]`

Tensor: implemented for integer and booleans tensors. 
As bitwise operation for integer tensors.
```python
import torch

~torch.tensor([1, 2, 3, False, True], dtype=torch.int32)
```
`tensor([-2, -3, -4, -1, -2], dtype=torch.int32)`

As logical negation for booleans tensors.
```python
import torch

~torch.tensor([False, True], dtype=torch.bool)
```
`tensor([ True, False])`

Numpy: Same bahavior as PyTorch.

Python lists: ⛔ It will raise a `TypeError` exception.

Pandas DataFrames: Same bahavior as PyTorch and numpy.

---
10.   What is `ProcessPoolExecutor`?
---
It's a class to execute callables asynchronously in separate processes using the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing) package. 

---
11.  How does `L.range(self.ds)` work?
---
It returns an `L` class object of size len(self.ds).

---
12. What is __iter__?
---
It's the special python method that indicates how an iterator should generate values while iterating. An `iterator` must have this method and the `__next__` method implemented.

---
13. What is first?
---
[first](https://fastcore.fast.ai/basics.html#first) is a fastcore function that returns the first element of an iterable x or None is its length is cero.

---
14. What is permute? Why is it needed?
---
It rearranges the dimensions of a tensor. It's needed for the images loaded using `PIL` as it saves the data `HWC` and PyTorch needs it in `HCW`.  

---
15. What is a recursive function? How does it help us define the parameters method?
---
It's a function that calls itself. It's used in this case to count the parameters of a `Module` including its children parameters.

---
16.  Write a recursive function that returns the first 20 items of the Fibonacci sequence.
---
NA

---
17. What is `super`?
---
`super()` returns a temporary object of the superclass (parent class). This is useful, among other use cases, to reuse methods for the super class and swap superclasses without changing the code.

---
18. Why do subclasses of `Module` need to override forward instead of defining `__call__`?
---
Because in the class `Module`, we are calling the function `forward` in `__call__`. This allows to have a generic function which makes sure of executing hooks.

---
19. In `ConvLayer`, why does `init` depend on `act`?
---
Because based on `act`, we are initializing the weights.

---
20. Why does `Sequential` need to call `register_modules`?
---
Because registering the modules will add the layers as `children` which is needed to include the parameter's layer in the parameters of the module. 

---
21. Write a hook that prints the shape of every layer's activations.
---
NA

---
22. What is "LogSumExp"?
---
It is the log of the sum of the exponentials.

$LSE(x_1​,…,x_N​)=\log(\sum_{n=1}^{n}{e^{n}})$

Using the [LogSumExp trick](http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/), we can calculate this value avoiding underflow or overflow errors.

---
23. Why is `log_softmax` useful?
---
As we are calculating the negative log likelihood and the `nll` doesn't include log, we calculate it with the softmax. 

---
24. What is `GetAttr`? How is it helpful for callbacks?
---
`GetAttr` is a fastai class that implements `__getattr__` (method called if an `AttributeError` exception is raised) and `__dir__` (method called to get the attributes of a class object when using `dir`) which returns a default attribute if the required is not found. This makes calling attributes from another object easier. In the callbacks case, we can call any attribute from `Learner` if it's the default.

---
25. Reimplement one of the callbacks in this chapter without inheriting from `Callback` or `GetAttr`.
---
NA

---
26. What does `Learner.__call__` do?
---
It executes the method `name` for every callback `cb` using the `getattr` method. If the attribute `name` doesn't exist it calls [noop](https://fastai1.fast.ai/core.html#noop).

---
27. What is `getattr`? (Note the case difference to `GetAttr`!)
---
`getattr` is a standard python function that returns an attribute from a given object. It's useful to execute a method with its name as parameter, which is helpful to execute a callback.

---
28. Why is there a try block in fit?
---
It's added to have the option to cancel the training by raising the exception `CancelFitException`.

---
29.  Why do we check for `model.training` in `one_batch`?
---
Because `one_batch` is called from `once_epoch` which is used for training and validation according to the value set in `model.training`.

---
30. What is `store_attr`?
---
It's a fastcore method that helps to assign multiple class attributes from comma separated names.

---
31. What is the purpose of `TrackResults.before_epoch`?
---
It initializes the parameters where the results of each epoch will be saved.

---
32. What does `model.cuda` do? How does it work?
---
It moves the model parameters to the GPU. 

---
33. Why do we need to check `model.training` in `LRFinder` and `OneCycle`?
---
Because these callbacks should be executed only when the model is being trained.

---
34. Use cosine annealing in `OneCycle`.
---
NA


## Further Research

---
1. Write `resnet18` from scratch (refer to Chapter 14 as needed), and train it with the `Learner` in this chapter.
---
NA

---
2. Implement a `batchnorm` layer from scratch and use it in your `resnet18`.
---
NA

---
3. Write a `Mixup` callback for use in this chapter.
---
NA

---
4. Add momentum to `SGD`.
---
NA

---
5. Pick a few features that you're interested in from fastai (or any other library) and implement them in this chapter.
---
NA

---
6. Pick a research paper that's not yet implemented in fastai or PyTorch and implement it in this chapter.
    - Port it over to fastai.
    - Submit a pull request to fastai, or create your own extension module and release it.
    - Hint: you may find it helpful to use [nbdev](https://nbdev.fast.ai/) to create and deploy your package.
---
NA
