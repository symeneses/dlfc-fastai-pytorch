# Questionnaire: A Neural Net from the Foundations


---
1. Write the Python code to implement a single neuron.
---
```python
output = sum([x*w for x,w in zip(inputs,weights)]) + bias
```

---
2. Write the Python code to implement ReLU.
---
```python
def relu(x): return x if x >= 0 else 0
```

---
3. Write the Python code for a dense layer in terms of matrix multiplication.
---
```python
x @ w.t() + b
```

---
4. Write the Python code for a dense layer in plain Python (that is, with list comprehensions and functionality built into Python).
---
```python
y[i,j] = sum([a * b for a,b in zip(x[i,:],w[j,:])]) + b[j]
```

---
5. What is the "hidden size" of a layer?
---
It's the number the neurons in the layer.

---
6. What does the `t` method do in PyTorch?
---
It transposes the matrix.

---
7. Why is matrix multiplication written in plain Python very slow?
---
Python is not optimized for numeric operations. Plain Python requires many for loops to perform calculations over arrays. 
PyTorch basic operations are written in C++ (fast and low level language) and it is designed to be efficient for matrixes calculations. 

---
8. In `matmul`, why is ac==br?
---
Following the definition of a matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix, therefore, we have to validate this condition.

---
9.  In Jupyter Notebook, how do you measure the time taken for a single cell to execute?
---
Using the magic command `%time`.

---
10. What is "elementwise arithmetic"?
---
It includes operations that can be performed across multiples arrays or tensors, applying the operation to each element in the same position in each array.

---
11.  Write the PyTorch code to test whether every element of `a` is greater than the corresponding element of `b`.
---
```python
import torch

a = torch.Tensor([[1,2,3],[4,5,6]])
b = torch.Tensor([[2,2,2],[5,5,5]])
a > b
```
`tensor([[False, False,  True],
        [False, False,  True]])`

---
12. What is a `rank-0` tensor? How do you convert it to a plain Python data type?
---
It's a tensor with a single element. We can convert a `rank-0` tensor to a Python data type by using the `item()` method.

```python
import torch

torch.Tensor([1]).item()
```
`1`

---
13. What does this return, and why? tensor([1,2]) + tensor([1])
---
```python
import torch

torch.Tensor([1,2]) + torch.Tensor([1])
```
`tensor([2., 3.])`

The result is summing the two tensor, as the second has a lower shape, it is broadcasted to the first. It's possible to broadcast these two vectors as the second is a scalar (1x1 tensor). 

---
14.  What does this return, and why? tensor([1,2]) + tensor([1,2,3])
---
```python
import torch

torch.Tensor([1,2]) + torch.Tensor([1,2,3])
```
`RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0`

It's not possible to broadcast in this case, as their shapes ([2] and [3]) in the same dimension are different.

---
15.  How does elementwise arithmetic help us speed up `matmul`?
---
Because we don't need to loop over the entire matrix to multiply values in the same position.

---
16.  What are the broadcasting rules?
---
- Dimensions are compaired from the right to the left, adding 1 to fill in for empty dimensions.
- The dimensions of two tensors are compatible if they are equal or one of them is 1.

---
17.  What is `expand_as`? Show an example of how it can be used to match the results of broadcasting.
---
`expand_as` is a method to expand the shape of a tensor to match the shape of another tensor applying the broadcasting rules.

```python
import torch

a = torch.Tensor([[1,2,3],[4,5,6]])
b = torch.Tensor([1,2,3])
b.expand_as(a)
```
`tensor([[1., 2., 3.], [1., 2., 3.]])`

---
18. How does `unsqueeze` help us to solve certain broadcasting problems?
---
`unsqueeze` creates a dimension of size 1 at the specified position, helping as to select the dimension to be broadcasted.

---
19. How can we use indexing to do the same operation as `unsqueeze`?
---
Using `None` as index in the dimension to be inserted.
```python
import torch

a = torch.Tensor([1,2,3])
a.unsqueeze(0).shape == a[None, :].shape
```

---
20.  How do we show the actual contents of the memory used for a tensor?
---
Seeing the corresponding `torch.Storage`, which is a one-dimensional array with the data of the tensor.
```python
import torch

a = torch.Tensor([[1,2,3],[4,5,6]])
b = torch.Tensor([1,2,3])
b.expand_as(a)
b.storage()
```
 `1.0
 2.0
 3.0
[torch.FloatStorage of size 3]`

---
21. When adding a vector of size 3 to a matrix of size 3×3, are the elements of the vector added to each row or each column of the matrix? (Be sure to check your answer by running this code in a notebook.)
---
The elements of the vector are added to each row of the matrix.
```python
import torch

v = torch.Tensor([10,20,30])
m = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]])
v + m
```
`tensor([[11., 22., 33.], [14., 25., 36.], [17., 28., 39.]])`

---
22. Do broadcasting and `expand_as` result in increased memory use? Why or why not?
---
No, because the storage shape of the tensor doesn't increase. The expanded or broadcasted tensor is built using the same data as the original with a `stride` (elements to be skipped in the storage to get the values in next dimension) which contains zero values.

---
23. Implement `matmul` using Einstein summation.
---
```python
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```

---
24. What does a repeated index letter represent on the left-hand side of einsum?
---
That we should sum over that index.

---
25. What are the three rules of Einstein summation notation? Why?
---
- Repeated indices on the left side are implicitly summed over if they are not on the right side.
- Each index can appear at most twice on the left side.
- The unrepeated indices on the left side must appear on the right side.

---
26. What are the forward pass and backward pass of a neural network?
---
The forward pass is when the output of the model is calculated. With the output, the loss and its gradients are calculated. The backward pass is the calculation of these gradients.

---
27. Why do we need to store some of the activations calculated for intermediate layers in the forward pass?
---
Because we have to backpropagate the loss function through all layers.

---
28. What is the downside of having activations with a standard deviation too far away from 1?
---
If the standard deviation is too high the gradients will explode, which means, after some layer, we will get infinite values.

---
29. How can weight initialization help avoid this problem?
---
We have to initialize the weights using a proper scale, which is found multiplying the weights by $1/\sqrt{n_i}$ where $n_i$ is the number of inputs in the layer.

---
30. What is the formula to initialize weights such that we get a standard deviation of 1 for a plain linear layer, and for a linear layer followed by ReLU?
---
- For a linear layer:
$$W_n(0) = \sqrt{\frac{1}{n_{in}}}W_n(0)$$

For a linear layer followed by ReLU (*He Initialization*):
$$W_n(0) = \sqrt{\frac{2}{n_{in}}}W_n(0)$$

---
31.  Why do we sometimes have to use the squeeze method in loss functions?
---
The output of the model has sometimes two dimensions while the targets are normally given as a vector. To calculate their difference, we have to squeeze the model output.

---
32. What does the argument to the squeeze method do? Why might it be important to include this argument, even though PyTorch does not require it?
---
The argument is the dimension of the tensor to be squeezed (removed). If not given, all dimensions of size 1 will be removed, which may cause unexpected results.

---
33. What is the "chain rule"? Show the equation in either of the two forms presented in this chapter.
---
It's the way of calculating the derivate of a composed function, in which the derivatives of the main or outer function are chained to the derivatives of the inner function.
$$F'(x) = (f\circ g)'(x) = f'(g(x))g'(x) $$

---
34. Show how to calculate the gradients of mse(lin(l2, w2, b2), y) using the chain rule.
---
$$\frac{\partial mse(lin(l_2, w_2, b_2), y)}{\partial l_2}  = \frac{\partial mse(lin(l_2, w_2, b_2), y)}{\partial lin(l_2, w_2, b_2)}\frac{\partial lin(l_2, w_2, b_2)}{\partial l_2}$$

---
35. What is the gradient of ReLU? Show it in math or code. (You shouldn't need to commit this to memory—try to figure it using your knowledge of the shape of the function.)
---
Given that:
$$ReLU(x) = \left\{ \begin{array}{cc}
{0} & {for} & x<=0 \\ 
x & {for} & x>0 \\\end{array}\right.$$
Then:
$$ReLU'(x) = \left\{ \begin{array}{cc}
{0} & {for} & x<=0 \\ 
1 & {for} & x>0 \\\end{array}\right.$$

---
36. In what order do we need to call the *_grad functions in the backward pass? Why?
---
In the reverse order as the * function was executed in the forward pass, because we have to apply the chain rule. First, calculating the outer function and then the inner function, and so forth, till reaching a non-composite function.

---
37. What is `__call__`?
---
`__call__` is a python special method which allows to create function-like objects by implementing the it in its respective class.

---
38. What methods must we implement when writing a `torch.autograd.Function`?
---
The `forward` (performs an operation) and `backward` (calculates the derivatives of the operation) methods. 

---
39.  Write `nn.Linear` from scratch, and test it works.
---
NA

---
40. What is the difference between `nn.Module` and fastai's `Module`?
---
The fastai's `Module` class calls the superclass `__init__` by default. For the PyTorch `nn.Module`, it has to be called when the class is initialized.


## Further Research

---
1. Implement ReLU as a `torch.autograd.Function `and train a model with it.
---
NA

---
2. If you are mathematically inclined, find out what the gradients of a linear layer are in mathematical notation. Map that to the implementation we saw in this chapter.
---
NA

---
3. Learn about the unfold method in PyTorch, and use it along with matrix multiplication to implement your own 2D convolution function. Then train a CNN that uses it.
---
NA

---
4. Implement everything in this chapter using NumPy instead of PyTorch.
---
NA
