# Questionnaire: CNN Interpretation with CAM


---
1. What is a "hook" in PyTorch?
---
A hook is a piece of code that can be injected in the forward or backward calculation of any layer.

---
2. Which layer does CAM use the outputs of?
---
Of the last convolutional layer.

---
3. Why does CAM require a hook?
---
To get the activations, we need to save the values once they are calculated.

---
4. Look at the source code of the `ActivationStats` class and see how it uses hooks.
---
NA

---
5. Write a hook that stores the activations of a given layer in a model (without peeking, if possible).
---
NA

---
6. Why do we call `eval` before getting the activations? Why do we use `no_grad`?
---
`eval` sets the model in evaluation mode, which means `train` is `False` which modifies the behaviour of some layers. `no_grad` makes sure PyTorch is not calculating gradients. 
We have to use them as we don't want to modify the model while getting the activations.

---
7. Use `torch.einsum` to compute the "dog" or "cat" score of each of the locations in the last activation of the body of the model.
---
NA

---
8. How do you check which order the categories are in (i.e., the correspondence of index->category)?
---
```py
dls.vocab
```

---
9.  Why are we using decode when displaying the input image?
---
To get the original values of the image, because the `DataLoader` applies normalization and any other given transformation.

---
10.  What is a "context manager"? What special methods need to be defined to create one?
---
It's an object that can be called with a `with` statement, which are not in usable state after the code inside the `with` are executed. To create a `context manager`, it's needed to implement the methods `__enter__()` and `__exit__()` which are invoked on the entrance to and exit from the body of the `with` statement.   

---
11.  Why can't we use plain CAM for the inner layers of a network?
---
Because only in the last layer, the gradients of the output with respect to the input are equal to the weights as this layer is lineal.

---
12.  Why do we need to register a hook on the backward pass in order to do Grad-CAM?
---
Because in the backward pass, it is when we are calculating the gradients, which are the ones used for `Grad-CAM`.

---
13. Why can't we call `output.backward()` when output is a rank-2 tensor of output activations per image per class?
---
Because gradients have to be calculated with respect to a value. For training, that value is the loss. For Grad-CAM, we need to select an image and a class, to get a unique value and then calculate the gradient: `output[image_number, class].backward() `.


## Further Research

---
1. Try removing `keepdim` and see what happens. Look up this parameter in the PyTorch docs. Why do we need it in this notebook?
---
NA

---
2. Create a notebook like this one, but for NLP, and use it to find which words in a movie review are most significant in assessing the sentiment of a particular movie review.
---
NA
