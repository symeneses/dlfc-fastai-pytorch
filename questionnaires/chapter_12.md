# Questionnaire: A Language Model from Scratch

---
1. If the dataset for your project is so big and complicated that working with it takes a significant amount of time, what should you do?
---
It's recommended to take a sample of the data to be able to iterate faster. For language modelling specifically, we can create simple datasets while we are in the analysis phase or prototyping.

---
2. Why do we concatenate the documents in our dataset before creating a language model?
---
To train a language model, we need sequences of text not a given document. We should in any case mark the end and start of a document so that the language model can learn to identify them.

---
3. To use a standard fully connected network to predict the fourth word given the previous three words, what two tweaks do we need to make to our model?
---
- **Set the activations that every layer will use**: The first linear layer should use only the first word's embedding, the second the second word's embedding plus the first layer's output activations and so on.

- **Share weight matrix across layers**: This avoids the model to learn the context of the words irrespective of its position. 

---
4. How can we share a weight matrix across multiple layers in PyTorch?
---
Only one layer is created and is reused whenever needed.

---
5. Write a module that predicts the third word given the previous two words of a sentence, without peeking.
---
NA

---
6. What is a recurrent neural network?
---
It's a multilayer neural network whose activations are calculated going through a sequence of tensors in a for loop. These tensors can be words, time series values, etc. 

---
7. What is "hidden state"?
---
The hidden state are the activations calculated at each step.

---
8. What is the equivalent of hidden state in `LMModel1`?
---
The linear layer `h_h`.

---
9.  To maintain the state in an RNN, why is it important to pass the text to the model in order?
---
Because the original text is split in shorter sequences to fit them in a batch. If the text is not in ordered, the model won't be able to see a complete long document.

---
10. What is an "unrolled" representation of an RNN?
---
It is a representation of the RNN where every layer corresponds to an item in the input sequence.

---
11. Why can maintaining the hidden state in an RNN lead to memory and performance problems? How do we fix this problem?
---
Maintaining the state implies to create a network with the size of the whole input, in that case, the GPU would need to keep in memory the gradient history, which is, practically speaking, not possible.

---
12.  What is "BPTT"?
---
Backpropagation Through Time is the approach of having one layer per time step calculated with the past n tokens.

---
13. Write code to print out the first few batches of the validation set, including converting the token IDs back into English strings, as we showed for batches of IMDb data in Chapter 10.
---
NA

---
14. What does the `ModelResetter` callback do? Why do we need it?
---
It initializes the activations of the model. As the RNN is stateful, we should reset the model to retrain it.

---
15.  What are the downsides of predicting just one output word for each three input words?
---
We are wasting information that can be used to update the weights of the model.

---
16. Why do we need a custom loss function for `LMModel4`?
---
To flatten the output first as it has the shape `bs (batch size) x sl (sequence length) x vocab_sz (vocab size)` and it should be `bs (batch size) x sl (sequence length)`.

---
17.  Why is the training of `LMModel4` unstable?
---
Because it has a deep network which depending on the random initial weights can lead to very small (`vanishing`) or very big (`exploding`) gradients.

---
18. In the unrolled representation, we can see that a recurrent neural network actually has many layers. So why do we need to stack RNNs to get better results?
---
Stacking multiple layers helps to find more complex relations between an input and its output.

---
19. Draw a representation of a stacked (multilayer) RNN.
---
NA

---
20. Why should we get better results in an RNN if we call detach less often? Why might this not happen in practice with a simple RNN?
---
In that case, the model will remember the gradients for a longer time. In practice, this is not feasible as we can't keep so much information in the GPU memory.

---
21. Why can a deep network result in very large or very small activations? Why does this matter?
---
A deep NN works multiplying matrices multiple times. If a small number is multiply by itself may times, it will result in an ever small number (the opposite happens with large numbers). This was an impediment to train NN as the model will reach a point where the wights are not being updated anymore or take infinite values.

---
22. In a computer's floating-point representation of numbers, which numbers are the most precise?
---
Numbers close to zero.

---
23. Why do vanishing gradients prevent training?
---
If the gradients are too small, the weights will be constant.

---
24. Why does it help to have two hidden states in the LSTM architecture? What is the purpose of each one?
---
One hidden state will keep the `cell state` or the long short-term memory and the other will be in charge of the actual prediction.

---
25. What are these two states called in an LSTM?
---
LSTM stands for Long Short-Term Memory as the main idea of this cells architecture is too retain memory of the given sequence.

---
26. What is `tanh`, and how is it related to sigmoid?
---
`Tanh` is a sigmoid function scaled to give values between -1 and 1.

$$tanh(x) = 2\sigma(x) - 1$$

---
27. What is the purpose of this code in LSTMCell: `h = torch.cat([h, input], dim=1)`
---
It joins the old hidden state and the input to create a big tensor with the vectors concatenated.

---
28.  What does chunk do in PyTorch?
---
It splits a tensor in a specific number of chunks which are views of the input tensor.

---
29. Study the refactored version of `LSTMCell` carefully to ensure you understand how and why it does the same thing as the non-refactored version.
---
NA

---
30. Why can we use a higher learning rate for `LMModel6`?
---
The LSTM architecture helps to control exploding gradients as the `cell gate` doesn't change much at every step and tanh function is applied to the `output gate`.

---
31.  What are the three regularization techniques used in an `AWD-LSTM` model?
---
- Dropout
- Activation regularization
- Temporal Activation regularization

---
32. What is "dropout"?
---
It's a regularization technique which assigns a probability `p` to a neuron to be off during a training iteration. The purpose is to make the NN more resilient and flexible. If not all cells are working at the same time, the NN will be forced to predict the correct output even if there is noise. A dropout rate of 1 means no dropout.

---
33.  Why do we scale the acitvations with dropout? Is this applied during training, inference, or both?
---
As the numbers of activations will vary, we need to normalize the activation by dividing them by `1 - p` ONLY during training. 

---
34. What is the purpose of this line from Dropout: `if not self.training: return x`
---
To be sure the dropout and its normalization is calculated only while training.

---
35. Experiment with bernoulli_ to understand how it works.
---
NA

---
36. How do you set your model in training mode in PyTorch? In evaluation mode?
---
When the function `train` of a module is called the module will set `training` as True and False when calling `eval`.

---
37. Write the equation for activation regularization (in math or code, as you prefer). How is it different from weight decay?
---
```py
loss += alpha * activations.pow(2).mean()
```
The weight decay penalizes large weights, while AR is regularizing the activations adding them to the loss function after multiplying them by an hyperparameter `alpha`.

---
38. Write the equation for temporal activation regularization (in math or code, as you prefer). Why wouldn't we use this for computer vision problems?
---
```py
loss += beta * (activations[:,1:] - activations[:,:-1]).pow(2).mean()
```
As LSTM is used for sequences, consecutive activations should be close. TAR penalizes differences between them. In computer vision, this assumption doesn't hold as images can change abruptly at any pixel.

---
39.  What is "weight tying" in a language model?
---
It's assigning the same values to the input embeddings `i_h` to the output hidden `h_o` layer. This can be done as both represent mappings from human language to activations.


# Further Research

---
1. In LMModel2, why can forward start with h=0? Why don't we need to say h=torch.zeros(...)?
---
NA

---
2. Write the code for an LSTM from scratch (you may refer to <
---
NA

---
3. Search the internet for the GRU architecture and implement it from scratch, and try training a model. See if you can get results similar to those we saw in this chapter. Compare your results to the results of PyTorch's built in GRU module.
---
NA

---
4. Take a look at the source code for AWD-LSTM in fastai, and try to map each of the lines of code to the concepts shown in this chapter.
---
NA
