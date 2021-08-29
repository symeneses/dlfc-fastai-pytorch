# Questionnaire: The Training Process


---
1. What is the equation for a step of SGD, in math or code (as you prefer)?
---
$$w -= \nabla w*lr$$

Where
- $w = Weights$
- $lr = Learning\ rate$

In code:
```py
w -= w.grad*lr
```

In PyTorch, we can use the [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) as shown in the example.
```py
# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# set the gradients to zero
optimizer.zero_grad()
# compute the loss
loss_fn(model(input), target).backward()
# perform an optimization step
optimizer.step()
```

We can use any other optimizer available in the module [torch.optim](https://pytorch.org/docs/stable/optim.html) under the section **Algorithms**.

---
2. What do we pass to `cnn_learner` to use a non-default optimizer?
---
We set the optimizer with the parameter `opt_func`.

---
3. What are optimizer callbacks?
---
Optimizer callbacks are functions that are used to modify an optimizer and execute an optimization step. They are called by the [Optimizer](https://docs.fast.ai/optimizer.html#Optimizer) class. 

---
4. What does `zero_grad` do in an optimizer?
---
It detaches [torch.Tensor.detach_](https://pytorch.org/docs/stable/generated/torch.Tensor.detach_.html) the tensor from the current graph and sets the gradients to zero as the standard PyTorch API.

---
5. What does step do in an optimizer? How is it implemented in the general optimizer?
---
It goes through a for loop through the callback and updates the state if values are returned from the callback.

---
6. Rewrite `sgd_cb` to use the += operator, instead of `add_`.
---
```py
def sgd_cb(p, lr, **kwargs): p.data += -lr*p.grad.data
```

---
7. What is "momentum"? Write out the equation.
---
Momentum is based on the idea of taking the moving average of the gradients as a direction to move in the next step. 

$$\overline{w} = \beta*\overline{w} +  (1 - \beta)*\nabla w$$
$$w =  w - lr*\overline{w}$$

Where
- $w = Weights$
- $lr = Learning\ rate$
- $\beta = Momentum$

---
8. What's a physical analogy for momentum? How does it apply in our model training settings?
---
Rolling a ball down a mountain, if the ball is bigger, it has more momentum and can jump out of small holes. In the model training settings, a higher $\beta$ will be less impacted by noise.

---
9.  What does a bigger value for momentum do to the gradients?
---
We will move in a similar direction always, missing small variations in the gradients.

---
10. What are the default values of momentum for `1cycle` training?
---
0.95 in the beginning and adjusted progressively to 0.85, and then moved back to 0.95 at the end of the training. As we read in the paper [A disciplined approach to neural network hyper-parameters: Part 1--learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820.pdf) in chapter 5, all forms of regularizations must be balanced.

---
11. What is RMSProp? Write out the equation.
---
RMSProp adjust the learning rate to every parameter, the ones which are unstable need a smaller learning rate while the ones close to zero need a larger learning rate to find an optimal value.

$$\overline{w^2} = \alpha*\overline{w^2} +  (1 - \alpha)*(\nabla w)^2$$
$$w =  w - \frac{lr*\nabla w}{\sqrt{\overline{w^2} + \epsilon}}$$

Where
- $w = Weights$
- $lr = Learning\ rate$
- $\alpha = Square\ momentum$

---
12. What do the squared values of the gradients indicate?
---
They indicate the dimension when the loss function is steeper. 

---
13. How does Adam differ from momentum and RMSProp?
---
Adam (Adaptive moment estimation) is a combination of momentum and RMSProp. It calculates the moving average of the gradients to get a direction and the exponentially decaying average of the squared gradients to adapt the LR to each parameter.

---
14. Write out the equation for Adam.
---
$$\overline{w} = \beta1*\overline{w} +  (1 - \beta1)*(\nabla w)$$
$$\overline{w} = \frac{\overline{w}}{(1 - \beta1^{(i+1)})}$$
$$\overline{w^2} = \beta2*\overline{w^2} +  (1 - \beta2)*(\nabla w)^2$$
$$w =  w - \frac{lr*\overline{w}}{\sqrt{\overline{w^2} + \epsilon}}$$

Where
- $w = Weights$
- $lr = Learning\ rate$
- $\alpha = Square\ momentum$

---
15. Calculate the values of `unbias_avg` and `w.avg` for a few batches of dummy values.
---
NA

---
16. What's the impact of having a high eps in Adam?
---
It will control the maximum value that the adjusted learning rate can take.

---
17. Read through the optimizer notebook in fastai's repo, and execute it.
---
NA

---
18. In what situations do dynamic learning rate methods like Adam change the behavior of weight decay?
---
When weight decay is applied as `L2 Regularization`, which is equivalent to weight decay for SGD, but not for adaptive gradient algorithms, the performance of Adam can be inferior. This is because calculating the gradients for Adam requires additional steps, so the weight decay must be applied only in the final step, when the new weights are being calculated.

---
19.   What are the four steps of a training loop?
---
1. Model
2. Loss
3. Gradients
4. Step

---
20.  Why is using callbacks better than writing a new training loop for each tweak you want to add?
---
It's difficult to adapt pieces of code to work together, using callbacks we are keeping the standard training loop and being flexible to add new functionalities knowing that they will be compatible with the rest of the code.

---
21. What aspects of the design of fastai's callback system make it as flexible as copying and pasting bits of code?
---
The fastai callback system has been successful because 1) it keeps the information about the batch, epoch and the training is available and 2) it allows the callback to fully control the training loop.

---
22.  How can you get the list of events available to you when writing a callback?
---
We can see it in the variable `event`.

```python
from fastai.vision.all import event

[e for e in dir(event) if not e.startswith("_")]
```

---
23. Write the ModelResetter callback (without peeking).
---
NA

---
24. How can you access the necessary attributes of the training loop inside a callback? When can you use or not use the shortcuts that go with them?
---
To access attributes, we can use `self`, for instance: `self.model`, `self.pred`, etc. To write them, we have to use `self.learn`. 

---
25. How can a callback influence the control flow of the training loop?
---
Raising excpetions, we can skip epochs `CancelEpochException` or batches `CancelBatchException` or stop the training `CancelFitException`, etc. Cancelling events (`after_cancel_batch`, `after_cancel_epoch`, `after_cancel_fit`, `after_cancel_step`, `after_cancel_train` `after_cancel_validate`) can be used to create callbacks that execute any action after any of these exceptions are raised.

---
26.  Write the `TerminateOnNaN` callback (without peeking, if possible).
---
NA

---
27. How do you make sure your callback runs after or before another callback?
---
The `Callback` class has a `run_after` and `run_before` variables to specify preceding or subsequent callbacks. 


## Further Research

---
1. Look up the "Rectified Adam" paper, implement it using the general optimizer framework, and try it out. Search for other recent optimizers that work well in practice, and pick one to implement.
---
NA

---
2. Look at the mixed-precision callback with the documentation. Try to understand what each event and line of code does.
---
NA

---
3. Implement your own version of the learning rate finder from scratch. Compare it with fastai's version.
---
NA

---
4. Look at the source code of the callbacks that ship with fastai. See if you can find one that's similar to what you're looking to do, to get some inspiration.
---
NA
