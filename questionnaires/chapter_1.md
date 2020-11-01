## Questionnaire: Your first Models
---

1. Do you need these for deep learning?
---

- Lots of math T / `F`
- Lots of data T / `F`
- Lots of expensive computers T / `F`
- A PhD T / `F`

To be a deep learning practitioner only high school Math is required, few samples and computing power is free in services like Google Colab.

---

2. Name five areas where deep learning is now the best in the world.
---

Few examples with links to their benchkmarks in [paperswithcode](https://paperswithcode.com).

- Computer visions: [Object detection](https://paperswithcode.com/task/object-detection)
- Speech: [Speech recognition](https://paperswithcode.com/task/speech-recognition)
- Natural Language Processing: [Machine translation](https://paperswithcode.com/task/machine-translation)
- Medicine: [Tumor segmentation](https://paperswithcode.com/task/tumor-segmentation)
- Playing games: [Go](https://paperswithcode.com/task/game-of-go)

---

3. What was the name of the first device that was based on the principle of the artificial neuron?
---

Mark I Perceptron which was a custom-built hardware to implement a perceptron algorithm designed for computer vision tasks.

---
4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?
---

In this book written in 1986, David Rumelhart and James McClellan proposed the requirements needed to make computers work similarly to a human brain.

- A set of processing units
- A state of activation
- An output function for each unit
- A pattern of connectivity among units
- A propagation rule for propagating patterns of activities through the network
- An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
- A learning rule whereby patterns of connectivity are modified by experience
- An environment within which the system must operate

---
5. What were the two theoretical misunderstandings that held back the field of neural networks?
---

Marvin Minsky and Seymour Papert demonstrated that a `perceptron` was unable to learn a simple function like an `XOR` (1980). They showed that adding a second layer could represent any mathematical function, but these networks were too big and too slow to perform practical applications. These discouraged the use of NN for some decades as explainable algorithms like Support Vector Machines (SVM) were preferred.

---
6. What is a GPU?
---

A graphics processing unit (GPU) is a circuit designed to process images, originally developed for gaming. Thanks to its parallel structure perform better for parallel processing needed to train deep learning algorithms.

---
7. Open a notebook and execute a cell containing: 1+1. What happens?
---

NA

---
8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
---

NA

---
9. Complete the Jupyter Notebook online appendix.
---

NA

---
10. Why is it hard to use a traditional computer program to recognize images in a photo?
---

We haven't understood completely how the process works, so we can not create a program which does it.

---
11. What did Samuel mean by "weight assignment"?
---

It's the process to assign values to parameters (variables) based on the performance so that it is optimized.

---
12. What term do we normally use in deep learning for what Samuel called "weights"?
---

Parameters, which can be weights but also variables to define a model architecture (e.g. hidden units, learning rate). Weights are the values updated through training.

---
13. Draw a picture that summarizes Samuel's view of a machine learning model.
---

```
[inputs + weights] -> model -> results
```
---
14. Why is it hard to understand why a deep learning model makes a particular prediction?
---

It is not easy to `interpret` what every neuron is doing. Interpreting the model needs to map every input to an output, which with a deep structure is not always feasible. It is some times possible to visualize which neurons are activated for a given input and then deduce what neurons are trying to do, but that's not always clear. 

---
15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?
---

`The universal approximation theorem` shows that a neural network can `approximate` any function and that higher number of hidden neurons improve the quality of its approximations.
For a very nice explanation, go [here!](http://neuralnetworksanddeeplearning.com/chap4.html)

---
16. What do you need in order to train a model?
---

A general mechanism that automatically update the weights. The mechanism used (for now) is the `Stochastic Gradient Descent (SGD)`.

---
17. How could a feedback loop impact the rollout of a predictive policing model?
---
Models predictions depends on historical data which can be biased. For example, a model to find a best fit for a job can give higher score to men, resulting in more men being hired, which will increase the bias in the data and therefore in the model. 

---
18. Do we always have to use 224×224-pixel images with the cat recognition model?
---
No, this is used in the example as old pretrained models has this input size. If you want to change it, the `ImageDataLoaders` has a parameter `item_tfms` used to transform every item. This can be used to resize the images with the transformation `Resize`.

---
19. What is the difference between classification and regression?
---
Classification is used to predict a `class` or `category` while `regression` tries to predict numeric values.

---
20. What is a validation set? What is a test set? Why do we need them?
---
The validation set is used to measure the accuracy or performance of the models on samples unseen by the model during training while tuning `hyperparameters` (number of layers, hidden units, etc). This is needed as the model can be `overfitting`, which means *memorizing* the training dataset. 
The test set is used for the same reason with the final model.

---
21. What will fastai do if you don't provide a validation set?
---
By default it will assign `valid_pct` to 0.2, which means 20% of the data will be used for validation.

---
22. Can we always use a random sample for a validation set? Why or why not?
---
No, there are some cases when this should be avoided.
- In classification, we want to split the data so that every class is represented (stratified sampling)
- For time series, we want to split the data based on time as the model shouldn't "see" the future
- In recommendation systems, the complete data of a user should be in the same dataset
 
---
23. What is overfitting? Provide an example.
---
When the model has almost an output for every input, including the noise but it doesn't generalize well, therefore it performs poorly with unseen data.

---
24. What is a metric? How does it differ from *"loss"*?
---
A metric evaluates the performance of the model. The main difference from loss is its purpose. The metrics is for us humans to compare the model, the loss function is used while training to update weights. For this reason, the loss function must be easy to derivate as that is required for the SGD to determine the direction in which weights should be updated.

---
25. How can pretrained models help?
---
Pretrained models save time and energy which is good for the environment as they were trained with a large dataset and can perform already some general task (ex. predicting the next word in a sentence). With a pretrained model, some layers stay frozen while some are fine-tuned using new data to customize the model for a specific use case (ex. classifying legal documents).

---
26. What is the *"head"* of a model?
---
The `head` is the one or more layers of the model which are fine tuned applying `transfer learning`.

---
27. What kinds of features do the early layers of a CNN find? How about the later layers?
---
The first layers are like filters that can detect features as edges, colors, shapes, etc. The last layers use the information gathered by the first layers to identify more complex patterns like textures, eyes or leaves.

---
28. Are image models only useful for photos?
---
No, they can be used also with spectrograms, medical data as X-Rays or computed tomography scan (CT scan), satellite imagery or videos. CNN can be used with data that is transformed to images as shown in the paper [Malware classification with deep convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/8328749/).

---
29. What is an *"architecture"*?
---
It's the mathematical function we are trying to fit to map the inputs to the output. With the architecture, we are including some assumptions about the model. If we decide to train a linear model, we are assuming the relation between the input and output is lineal.

---
30. What is segmentation?
---
Segmentation is to classify every pixel of an image to determine to which object it belongs. The model is trained to color-code every pixel, so that every pixel that is a cat will be of the color X, every car will be of color Y.

---
31. What is `y_range` used for? When do we need it?
---
It's used to set a range for the predictions of a continuos variable.

---
32. What are *"hyperparameters"*?
---
Hyperparameters are parameters that define the model behavior and normally are not updated during training. Every algorithm has its specific hyperparameters but there are some common for neural networks (ex. learning rate, number of layers).

---
33. What's the best way to avoid failures when using AI in an organization?
---
- Be sure you have the data! DL doesn't work without it.
- Follow a methodology that acknowledges the cyclic nature of a AI project like [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- Define business metrics to measure if your project brings value to the organization or not, measure this metric before the project for future comparisons
- Set a simple baseline and iterate over it. It's better to fail early, all models are wrong so you don't need a perfect one, you need one that improves your business metric and is better that a simple deadline

## Further Research

---
1. Why is a GPU useful for deep learning? How is a CPU different, and why is it less effective for deep learning?
---

---
2. Try to think of three areas where feedback loops might impact the use of machine learning. See if you can find documented examples of that happening in practice.
---

