## Questionnaire: From Model to Production

---
1. Where do text models currently have a major deficiency?
---
Text models currently are good to classify test, and text generation tasks as translation and summarization. However, the text generated, even though it looks like English or Chinese can contain false statements as the models don't really understand the meaning of a sentence or a word and they can't validate new information using previous knowledge.

---
2. What are possible negative societal implications of text generation models?
---
Fake news whose fact-checking takes a lot more time than needed to spread on social media

---
3. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
---
Give the last word to a human. The automation process can help us to prioritize cases or to assign a case to a specific resource.

---
4. What kind of tabular data is deep learning particularly good at?
---
DL provided some advantage over other machine learning algorithms for natural language, and categorical data with high cardinality as, in general, DL is good at encoding. 

---
5. What's a key downside of directly using a deep learning model for recommendation systems?
---
Recommendations systems are widely used and are the essence of many companies, but they still have many shortcomings. The recommendations given are based on the user and similar user's history, therefore, they fail to suggest something new or the suggestions given can be an item the user would've bought or watched without the recommendation. This applies, however, not only for systems built using DL.

---
6. What are the steps of the Drivetrain Approach?
---
- Defined objective: what do we want?
- Levers: what is in our control?
- Data: which data do we have for that?
- Models: how can we reach the goal controlling the levers?

---
7. How do the steps of the Drivetrain Approach map to a recommendation system?
---
- Defined objective: Increase the revenue of the company through cross-selling
- Levers: What the users see where they are buying
- Data: History of users' transactions, characteristics of our products
- Models: A/B test for a Recommender System

---
8. Create an image recognition model using data you curate, and deploy it on the web.
---
NA

---
9. What is `DataLoaders`?
---
It's a fastai class which loads the data to be used for training as `train` and `valid`. 

---
10. What four things do we need to tell `fastai` to create [DataLoaders](https://docs.fast.ai/data.core.html#DataLoaders)?
---
To create `DataLoaders` and `Datasets`, you may use a [DataBlock](https://docs.fast.ai/data.block#DataBlock).

- Type of data for inputs/labels: A [TransformBlock](https://docs.fast.ai/data.block#TransformBlock) in `blocks`
- How to get the items: function in `get_items`
- How to label every item: A [Split](https://docs.fast.ai/data.transforms#Split) function in `get_y`
- How to create the validation set: A `Splitter` in `splitter`

---
11. What does the `splitter` parameter to `DataBlock` do?
---
In this parameters, we set a function that will be used to split the items in the training and validation sets.

---
12. How do we ensure a random split always gives the same validation set?
---
Setting a seed in the split function so that the random numbers generation will start always in the same place.

---
13. What letters are often used to signify the independent and dependent variables?
---
$y = f(x)$
- $x: independent$
- $y: dependent$

---
14. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
---
- Crop: It fits a square (normally or the size we use as input) using the width or the height, whichever is smaller.
- Pad: It fits the image to the input shape using the width or the height, whichever is bigger and filling the rest with zeros (black).
- Squish: Resize the whole image to fit the given input shape.

---
15.    What is data augmentation? Why is it needed?
---
It's a technique to enrich the training dataset by applying different random transformations to every image. This helps to increase the variability of the input data with images in different sizes, angles, colors, brightness, contrast, etc.

---
16.   Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
---
An image with a group of bears aka sloth.

---
17.  What is the difference between `item_tfms` and `batch_tfms`?
---
`item_tfms` is used to apply transformations to every item and `batch_tfms` to a batch.

---
18.  What is a confusion matrix?
---
A confusion matrix shows the number counts of `actual` labels vs the `predicted` labels. In the diagonal then we have everything that was correctly classified and the misclassifications everywhere else.

---
19.  What does `export` save?
---
The architecture and parameters of the model in a pickle file, which is a serialized python object.

---
20.  What is it called when we use a model for getting predictions, instead of training?
---
Inference

---
21.  What are IPython widgets?
---
GUI components that allow us to create web applications from a Jupyter notebook.

---
22.  When might you want to use CPU for deployment? When might GPU be better?
---
CPUs are cheaper and if we are predicting only one sample at a time, we are not really using the GPUs capabilities. GPUs would be an option only if we can do predictions for batches.

---
23.  What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
---
You need to transfer the client data to the server always, so:
- The device needs internet connection
- You have to deal with some privacy issues
And you need to manage and scale the services according to the incoming requests.

---
24.   What are three examples of problems that could occur when rolling out a bear warning system in practice?
---
- The resolution of images captured by a camera is lower than the given for training => `Out-of-domain data`
- The positions bears are in the pictures are different to the ones when they are in movement => `Out-of-domain data`
- The prediction can be available when it's too late to take an action

---
25. What is `out-of-domain data`?
---
The data used for training has a different distribution or different patterns to the one the system is seeing in production.

---
26.   What is `domain shift`?
---
It's when the input data on production's systems changes over time, in this case, the original training data is not longer a good sample.

---
27.  What are the three steps in the deployment process?
---
- Manual process: Run the old process as usual while comparing with the model results to validate if the outputs given are correct. In this point, the model can be use as tool to help humans to prioritize or flag cases.
- Limited scope deployment: Deploy a model for a limited and controlled scope, this scope could be defined by location, time span, users, etc. In this step, human supervision is needed and we should have the option to go back to the manual process.
- Gradual expansion: Increase the scope of the rollout based on the established metrics. These metrics should consider all possible negative consequences and alert in case things are not as expected.

## Further Research

---
1. Consider how the **Drivetrain Approach** maps to a project or problem you're interested in.
---
NA

---
2. When might it be best to avoid certain types of data augmentation?
---
- Flipping images from a video game as it will create images that will never appear
- Altering the color or hue of medical data as skin marks
- For text, in general, apply data augmentation is not a good data

---
1. For a project you're interested in applying deep learning to consider the thought experiment "What would happen if it went really, really well?"
---
NA

---
4. Start a blog, and write your first blog post. For instance, write about what you think deep learning might be useful for in a domain you're interested in.
---
NA