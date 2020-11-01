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

---
17. How could a feedback loop impact the rollout of a predictive policing model?
---

---
18. Do we always have to use 224×224-pixel images with the cat recognition model?
---

---
19. What is the difference between classification and regression?
---

---
20. What is a validation set? What is a test set? Why do we need them?
---

---
21. What will fastai do if you don't provide a validation set?
---


## Further Research

---
1. Why is a GPU useful for deep learning? How is a CPU different, and why is it less effective for deep learning?
---

---
2. Try to think of three areas where feedback loops might impact the use of machine learning. See if you can find documented examples of that happening in practice.
---

