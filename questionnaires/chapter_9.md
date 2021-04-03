# Questionnaire: Tabular Modeling Deep Dive

---
1. What is a continuous variable?
---
It refers to numerical data that can take an infinite set of values as opposed to discrete variables, which can take values define on a specific range. As it's numerical, we can perform any linear operation on them.

---
2. What is a categorical variable?
---
A category variable contains can contain only a specific set of discrete values (eg. gender, zipcode). For modelling purposes, these variables need to be encoded in a numerical representation.

---
3. Provide two of the words that are used for the possible values of a categorical variable.
---
Sentiment = {0: Happiness, 1: Sadness}

---
4. What is a "dense layer"?
---
It's a linear layer a.k.a. fully connected layer as all inputs and outputs are connected. 

---
5. How do entity embeddings reduce memory usage and speed up neural networks?
---
One hot encoding requires n length size vectors, where n is the number of categories. Using embeddings, we can define a far lower vector size reducing the computations and the memory required to save intermediate results.

---
6. What kinds of datasets are entity embeddings especially useful for?
---
Datasets with high cardinality features (eg. zip codes) which tends to overfit using classical methods.

---
7. What are the two main families of machine learning algorithms?
---
- Ensembles of decision trees: Random Forest, Gradient Boosting Machines (GBM)
- Neural Networks: CNN, LSTM, Transformers, etc

---
8. Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?
---
Some categorical variables have an implicit ordering (eg. Education level, Hotel category)
```py
import pandas as pd

data = pd.DataFrame({"names": ["Super Fancy Hotel", "Cheap & Good", "Cheap", "Good", "Ok"], 
                     "stars": ["5", "2", "1", "4", "3"], 
                     "price": [1500, 400, 350, 1100, 900]})
data["stars"] = data["stars"].astype(pd.CategoricalDtype(ordered=True))
data["stars"].dtype
```

---
9. Summarize what a decision tree algorithm does.
---
One of the most common decision trees algorithm is the Classification And Regression Tree (CART). It splits the training set in two subsets using a feature `k` and a threshold `tk` so that the data in each group is as pure as possible.

To measure the quality, it can use:

- Gini -> 1 − Σp(i,k)^2 for each class k in the node i
- Entropy-> −Σp(i,k)*log(p(i,k)) for each class k in the node i

Both are zero if all datapoints belong to the same class.

Decision trees are non-parametric methods used for classification and regression. Trees can be visualized, work well with numerical and categorical data but they tend to overfit. Thanks to their simplicity, they are used for ensembling methods. 

---
10. Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow it to be used in a model?
---
First, a date contain different variables that can be used as features as they can help find patterns or seasonalities e.g. day of the week, week of the month, holiday, etc.
Second, while spliting the dataset into train/dev/test we have to be careful to not include future data in the dev of test set (it's easier to predict the future if you know it!).

---
11.  Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should you pick?
---
We need to split the dataset depending in the `saledata` column as this is a prediction task and the validation set should have only data with posterior dates to the training set.  

---
12. What is pickle and what is it useful for?
---
A pickel is a data format that save serialized Python objects using binary protocols. This means, that pickles files can be read only using python and when loaded they are python objects unlike JSON files which can be used in any platform.

---
13.  How are `mse`, `samples`, and `values` calculated in the decision tree drawn in this chapter?
---
- **mse**: It's the Mean Squared Error (MSE) if the **value** is predicted for all samples in the node.
- **samples**: It's the number of samples that meet the condition. In the top node before any split is the size of the training dataset.
- **values**: It's the predicted value for the samples that meet the condition. In the top node before any split is the average value of the dataset.

---
14.   How do we deal with outliers, before building a decision tree?
---
Decision trees are non-parametric algorithms, which means they don't make any assumptions about the data making them very robust to outliers. 

---
15.  How do we handle categorical variables in a decision tree?
---
In decision trees, there are not embedding layers to encode the meaning of categories. That leaves us with two options: 
- **One-hot encoding**: Every category is a one-hot encoded column. This increases the potential splits exponentially, making more difficult to work with the dataset.
- **Use the original categories**: The decision tree normally would create splits if a specific category has the capacity to classify or calculate the predictor.

---
16. What is bagging?
---
Bagging generates multiple training datasets using random sampling with replacement and trains in parallel different models which will vote to do a final prediction. Bagging helps to avoid overfitting.

[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) is a bagging method that randomized the features creating more diverse trees.

---
17. What is the difference between `max_samples` and `max_features` when creating a random forest?
---
- **max_samples**: sample size or number of rows used to train each decision tree
- **max_features**: it restricts the number of features or columns to be used to train each tree

---
18.  If you increase `n_estimators` to a very high value, can that lead to overfitting? Why or why not?
---
No, in most cases. Random Forests has many other parameters that help to avoid overfitting such as: `max_samples`, `max_features` , `min_samples_leaf` and `max_depth`. It has been shown that with more estimators using a subset of features, we can get hights levels of accuracy.

---
19. In the section "Creating a Random Forest", after Figure 9-7, why did `preds.mean(0)` give the same results as our random forest?
---
The RF by definition predicts using the mean of its estimators.

---
20. What is **"out-of-bag-error"**?
---
It's the error of every row respect to the prediction made by trees trained without that particular row.

---
21.  Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses?
---
- The distribution of data in the validation set is different.
- The model is overfitting.

---
22.  Explain why random forests are well suited to answering each of the following question:
---
- **How confident are we in our predictions using a particular row of data?** Random Forest estimates every row prediction taken an average of the predictions of its estimators. Calculating the standard deviation of these values, we can know how confident the prediction is, that is, the more agreement among trees the less standard deviation.
- **For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?** For each estimator and node, we can calculate how the prediction is changing and add up these changes as the contributions of the factor in the node condition. These can be done using the library `treeinterpreter` and visualized with `waterfallcharts`.
- **Which columns are the strongest predictors?** With RF, we can estimate how each column improves each tree and weight this improvement based on the samples in the node. These values after being normalized are used to know what features have more prediction power.
- **How do predictions vary as we vary these columns?** RF calculates the OOB score (value between `0: Random model` and `1: Perfect model` based on Out-of-Bag-Error). We can calculate this score using different columns and see if the value is being affected.

---
23.   What's the purpose of removing unimportant variables?
---
It's easier to improve, explain and deploy a model with less columns. This follows the [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor) *"entities should not be multiplied without necessity"*

---
24.  What's a good type of plot for showing tree interpreter results?
---
A waterfall plot as they help to see the marginal value contributions to a starting value. In our case, the tree interpreter gives a bias (starting point) and the contribution of each feature.

---
25.  What is the **"extrapolation problem"**?
---
It's the lack of generalization present in Machine Learning and Deep Learning algorithms as they can predict values outside the range seen while training. It can be observed if we try to predict using data that is not distributed equally as the training set a.k.a `out-of-domain data`.  

---
26. How can you tell if your test or validation set is distributed in a different way than your training set?
---
We can train a model to predict if a row belongs to the training or to the validation set. If the two sets are equally distributed, we shouldn't be able to get good results. If that's not the case, we can see which features that are able to identify the set and remove them if possible.

---
27. Why do we make `saleElapsed` a continuous variable, even although it has less than 9,000 distinct values?
---
Because it's a numerical variables by definition that encodes the number of days after the start of the dataset. If treated as category the model will not extrapolate outside the values given in the training set.

---
28.  What is "boosting"?
---
Boosting starts creating a simple model that can get good results for easy samples, and it gives bigger weights to samples that are difficult to classify. In the next iteration, it will create models focusing on those samples with higher weights. In that way, it optimizes to get higher accuracies. 

[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier) is the classical example of Boosting. AdaBoost predicts the class based on weighted votes, as each estimator (tree) has a weight depending on its accuracy. [Gradient Tree Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier) is similar, but uses the residuals errors from the last predictor and supports subsampling or `Stochastic Gradient Boosting`. With Stochastic Gradient Boosting a subsample, whose size is controlled with a parameter `subsample`, is taken to train each tree allowing `out-of-bag error` calculation.

---
29. How could we use embeddings with a random forest? Would we expect this to help?
---
Categorical variables can be encoded in embeddings training a NN first. The embeddings are posteriorly used to train any other model, this approach helps to improve accuracy as the embeddings keep important information about the categories e.g: close zipcodes will have close vectors.

---
30.  Why might we not always use a neural net for tabular modeling?
---
Training neural nets requires more time, data and computing resources. In general, it's recommended to start with Random Forest as a baseline and only if we can see improvements move to a more complicated and difficult to tune algorithm.


# Further Research

---
1. Pick a competition on Kaggle with tabular data (current or past) and try to adapt the techniques seen in this chapter to get the best possible results. Compare your results to the private leaderboard.
---
NA

---
2. Implement the decision tree algorithm in this chapter from scratch yourself, and try it on the dataset you used in the first exercise.
---
NA

---
3. Use the embeddings from the neural net in this chapter in a random forest, and see if you can improve on the random forest results we saw.
---
NA

---
4. Explain what each line of the source of `TabularModel` does (with the exception of the `BatchNorm1d` and `Dropout` layers).
---
Na
