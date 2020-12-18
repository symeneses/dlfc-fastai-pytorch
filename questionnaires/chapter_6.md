## Questionnaire: Other Computer Vision Problems

---
1. How could multi-label classification improve the usability of the bear classifier?
---
A multi-label classifier can be used to detect *multiple matches* (two different bears in the same image) or *zero matches* (no known bears in the picture). 

---
2. How do we encode the dependent variable in a multi-label classification problem?
---
The labels can't be given in path names as it's the case for single-label classifiers. The labels are given normally as an array in a file.
We encode the dependent variable using `one-hot` encoding, which convert the labels to a vector of 0's and 1's. Every label is represented in the vector, so the length of the vector is the number of unique labels and it will be 1 when the respective label is present in the item.

---
3. How do you access the rows and columns of a `DataFrame` as if it was a matrix?
---
In a pandas DataFrame, we can use the property `iloc` to filter rows and columns using their indexes.

---
4. How do you get a column by name from a DataFrame?
---
In a pandas DataFrame, we can index the name of the column to access it as a pandas `Series`.

---
5. What is the difference between a Dataset and DataLoader?
---
The `Dataset` is a collection of tuples (X: independent, y: dependent) and a `DataLoader` is an iterator that provides X and y mini-batches.

---
6. What does a `Datasets` object normally contain?
---
An iterator with a training and a validation `Dataset`

---
7. What does a `DataLoaders` object normally contain?
---
A training and a validation `DataLoader`

---
8. What does lambda do in Python?
---
```py
lambda arguments : expression
```
Lambda defines anonymous functions which are useful to create single expressions routines that are invoked only once in our code.

---
9.  What are the methods to customize how the independent and dependent variables are created with the data block API?
---
In the class `DataBlock`, the parameters `get_x` and `get_y` allow us to set functions that can transform the input data.

---
10. Why is `softmax` not an appropriate output activation function when using a one hot encoded target?
---
The softmax function makes sure all values sum up to 1, condition needed for a single label. For a one-hot encoded target, we shouldn't restrict this value as the goal is to find a flexible number of labels.

---
11. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
---
The `nll_loss` function will calculate values only for the target index which has to be a single integer.  

---
12.  What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
---
[BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) will calculate the sigmoid function nefore the applying the binary cross entropy function.

---
13. Why can't we use regular accuracy in a multi-label problem?
---
The regular accuracy assumes there is only one right label, therefore, it takes as prediction the label with the highest activation. To estimate multi-label predictions, we need to set a threshold so that activation values higher than this threshold are predicted as positive.

---
14. When is it okay to tune a hyperparameter on the validation set?
---
To select an hyperparameter, it's always a good practice to plot a metric vs the hyperparameter. If we observe that the hyperparameter gives good performance for a wide range, that means we are not overfitting or tuning values that will perform well specifically for the validation set. 

---
15. How is y_range implemented in fastai? (See if you can implement it yourself and test it without peeking!)
---
y_range uses the sigmoid function, which gives values between 0 and 1, times the range (maximum values - minimum value) plus the minimum value. With this function, we will be sure to get values within the given range as shown in the example. 
```py
import torch

def sigmoid_range(x, range=(0, 1)): 
  return (range[1] - range[0])*torch.sigmoid(x) + range[0]

x = torch.Tensor([-100] + list(range(-5, 5)) + [100])
sigmoid_range(x)
```
tensor([0.0000, 0.0067, 0.0180, 0.0474, 0.1192, 0.2689, 0.5000, 0.7311, 0.8808, 0.9526, 0.9820, 1.0000])

---
16.  What is a regression problem? What loss function should you use for such a problem?
---
Regression is a kind of problem where the output is numerical and continuos. In this case, the Mean Squared Error (`MSELoss` in PyTorch) is recommended as it penalizes bigger errors (target value - prediction).

---
17. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?
---
fastai will do that (isn't that cool?) if we define in the DataBlock the dependent variable as [PointBlock](https://docs.fast.ai/vision.data.html#PointBlock).


## Further Research

---
1. Read a tutorial about Pandas DataFrames and experiment with a few methods that look interesting to you. See the book's website for recommended tutorials.
---
NA

---
2. Retrain the bear classifier using multi-label classification. See if you can make it work effectively with images that don't contain any bears, including showing that information in the web application. Try an image with two different kinds of bears. Check whether the accuracy on the single-label dataset is impacted using multi-label classification.
---
NA