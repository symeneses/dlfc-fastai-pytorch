
# Questionnaire: Training a State-of-the-Art Model

---
1. What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?
---
Imagenette is a subset with 10 categories from ImageNet which has 1000 categories. Mostly, Imagenette is apt for experimenting as with less data, we can iterate faster.

---
2. What is normalization?
---
It's transforming the data so that its mean is 0 and standard deviation is 1. This is a important step while training a model from scratch as the statistics in every dataset are different. 

---
3. Why didn't we have to care about normalization when using a pretrained model?
---
With a pre-trained model, fastai will automatically normalized the data using the statistics of the original training data.

---
4. What is progressive resizing?
---
It's a technique to speed up training, starting with small images and gradually use larger images.

---
5. Implement progressive resizing in your own project. Did it help?
---
NA

---
6. What is test time augmentation? How do you use it in fastai?
---
Test Time Augmentation (TTA) is the use of data augmentation on the validation and test set. The prediction for each item is the average of the predictions for the respective augmented versions.
In fastai, by default in the method `Learner.tta`, a center crop image plus four randomly augmented versions are applied to the validation set. Customized transformations can be set in `item_tfms` or `batch_tfms`  and a given `DataLoader` ind `dl`.

---
7. Is using TTA at inference slower or faster than regular inference? Why?
---
Using TTA implies longer times as we need to transform the data and calculate a prediction for each defined transformation.

---
8. What is Mixup? How do you use it in fastai?
---
It's a data augmentation technique that creates new items with the weighted average of randomly selected pairs. 
$$new_items -= \lambda*item1 + (1 - \lambda)*item2$$

Where $\lambda$ is a random weight.

---
9. Why does Mixup prevent the model from being too confident?
---
With Mixup, labels are normally not 1s and 0s which helps to avoid the activations from taking extreme values.

---
10. Why does training with Mixup for five epochs end up worse than training without Mixup?
---
Mixup increases training times as in every epoch the model is seeing new images and has to predict two labels instead of one.

---
11. What is the idea behind label smoothing?
---
Softmax and sigmoid are 0 and 1 only for -inf and +inf, therefore, activations tend to larger numbers. This makes the model very rigid as the gradients are values between 0 and 1. With `label smoothing` labels are recalculated based on an $\epsilon$ which can be seen as our confidence on the labels:
$$y = \epsilon/N \ for \ y=0$$ 
$$y = 1 - \epsilon - \epsilon/N \ for \ y=1$$

With this transformations, there will be a large difference between 0s and 1s but the model won't be forcing activations to be larger and larger.

---
12.  What problems in your data can label smoothing help with?
---
Data can be mislabel. Label smoothing allow us to represent in the model our uncertainty about the labels and train a model which is not overconfident about its predictions.

---
13. When using label smoothing with five categories, what is the target associated with the index 1?
---
$$
y = 1 - \epsilon - \epsilon/5 = 0.92 \\
where \ \epsilon=0.1
$$

---
14. What is the first step to take when you want to prototype quick experiments on a new dataset?
---
To speed up experiments, it's recommended to create a representative subset or sample of the whole dataset.

## Further Research

---
1. Use the fastai documentation to build a function that crops an image to a square in each of the four corners, then implement a TTA method that averages the predictions on a center crop and those four crops. Did it help? Is it better than the TTA method of fastai?
---
NA

---
2. Find the Mixup paper on arXiv and read it. Pick one or two more recent articles introducing variants of Mixup and read them, then try to implement them on your problem.
---
NA

---
3. Find the script training Imagenette using Mixup and use it as an example to build a script for a long training on your own project. Execute it and see if it helps.
---
NA

---
4. Read the sidebar "Label Smoothing, the Paper", look at the relevant section of the original paper and see if you can follow it. Don't be afraid to ask for help!
---
NA