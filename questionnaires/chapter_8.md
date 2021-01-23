# Questionnaire: Training a State-of-the-Art Model

---
1. What problem does collaborative filtering solve?
---
Recommending useful products or products that users may like having a large set of users and products.

---
2. How does it solve it?
---
Recommending to the target user unseen products which have been seen by users who have common liked products in their history.

---
3. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?
---
Because it will show the user products that he may have found without any recommendation. Another problem, it's that creates a feedback loop where the user will always see the same kind of products.

---
4. What does a crosstab representation of collaborative filtering data look like?
---
The crosstab representation show a matrix of users vs products. Normally, this matrix will be very sparse as users sees only few products and each product is seen by few users.        

| user_id\product_id | 25 | 37 | 49 | 51 |
|--------------------|----|----|----|----|
| 9863               | 3  |    |    | 5  |
| 9864               |    | 1  |    |    |
| 9865               |    |    |    | 3  |

---
5. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).
---
```py
pd.crosstab(index=ratings['user'], columns=ratings['movie'], values=ratings['rating'], aggfunc=np.max) 
```

---
6. What is a latent factor? Why is it "latent"?
---
Latent is something hidden, unmeasured or not observed. In Statistics, latent variables are inferred from observable variables using a mathematical model. 
In collaborative filtering, the word factor is used as it's not possible to give a meaning to every vector component.

---
7. What is a dot product? Calculate a dot product manually using pure Python with lists.
---
The dot product is an operation between two vectors that gives a real number. This number represents the projection of one vector on the other.
The dot product is calculated by summing up the element-wise multiplication of two vectors.

$$\vec{u}.\vec{v} = \sum_{i=1}^{n} u_iv_i $$

```py
X = [1, 3, 5, 7]
Y = [2, 4, 6, 8]
# Equal to np.dot(X, Y)
dot_product = sum(i*j for (i, j) in zip(X, Y))
dot_product
```
100

---
8. What does `pandas.DataFrame.merge` do?
---
The merge method joins two DataFrames on columns (rows with the same value in the given columns) or indexes (rows with the same indexes).

For more details, see [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html).

---
9.  What is an embedding matrix?
---
It's a one-hot-encoded matrix used to index data with linear operations as $matrix^\intercal*one_hot$ is equal to $matrix[one_hot == 1]$. We can see this in the code snippet.

```py
import torch

users_factors = torch.randn(5, 5)
movie_factors = torch.randn(5, 5)
one_hot = torch.tensor([0.0, 0, 0, 1, 0])
users_factors.t() @ one_hot == users_factors[torch.argmax(one_hot, dim=0)]
```
tensor([True, True, True, True, True])

---
10.  What is the relationship between an embedding and a matrix of one-hot-encoded vectors?
---
The embedding is a layer that represents a one-hot-encoded vector which contains the indexes of ones in the encoded vector and its derivative are.   

```py
import torch

one_hot = torch.tensor([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0]])
embedding = torch.argmax(one_hot, dim=-1)
embedding
```
tensor([3, 1])

---
11.  Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?
---
With embeddings, we are saving memory and reducing processing times.

---
12. What does an embedding contain before we start training (assuming we're not using a pretained model)?
---
Random numbers.

---
13. Create a class (without peeking, if possible!) and use it.
---
NA

---
14. What does x[:,0] return?
---
The first column of the matrix x.

---
15. Rewrite the DotProduct class (without peeking, if possible!) and train a model with it.
---
NA

---
16. What is a good loss function to use for MovieLens? Why?
---
Mean Squared Error (MSE) as the dimensions between the output and targets are different and their data type is numerical.

---
17. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?
---
We would need to have equal dimensions for the targets and embeddings.

---
18. What is the use of bias in a dot product model?
---
It will create a representation of how good or bad a movie or user is. For example, movies with higher bias will be preferred by users irrespective of their specific preferences.

---
19. What is another name for weight decay?
---
L2 regularization as the sum of the weights squared is added to the loss function.

---
20. Write the equation for weight decay (without peeking!).
---
$$WD = \sum weights^2$$
This value multiple by a parameter *wd* is added to the loss.

---
21.  Write the equation for the gradient of weight decay. Why does it help reduce weights?
---
$\nabla WD=2*weights$

As this value is increasing the loss, the optimization process will try to reduce the weights.

---
22. Why does reducing weights lead to better generalization?
---
Reducing the weights is a technique to control overfitting. Small coefficients in a function will avoid abrupt changes in the function resulting in a smoother function.

---
23.  What does `argsort` do in PyTorch?
---
The [argsort](https://pytorch.org/docs/stable/generated/torch.argsort.html#torch-argsort) function returns the indices that sort a tensor along a given dimension in ascending order by value.

---
24. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?
---
No, the biases represents the deviation of the movie from similar movies. The average will show only the movies popularity in general.

---
25.  How do you print the names and details of the layers in a model?
---
```py
# PyToch
print(model)
# fastai learner
print(learner.model)
```

---
26. What is the "bootstrapping problem" in collaborative filtering?
---
It's the challenge of recommending products to new users and selecting users to recommend new products.

---
27. How could you deal with the bootstrapping problem for new users? For new movies?
---
For new users:
- Define an average taste (average vectors of all users) and recommend it by default.
- Create a tabular model (user data vs user embeddings), so that you can use user details to infer its embeddings.

For new movies:
- Create a tabular model (movie data vs movie embeddings), so that you can use the new movie details to infer its embeddings.

---
28.  How can feedback loops impact collaborative filtering systems?
---
More active users will have a higher impact on our recommendations, which will lead to attract more similar users.

---
29. When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?
---
The embeddings are concatenated, therefore, there are no operations among them.

---
30. Why is there an `nn.Sequential` in the CollabNN model?
---
Because we are defining the model as a sequence of layers, that is, starting from the input layer till the output.

---
31. What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?
---
A tabular model where the metadata are additional features.

## Further Research

---
1. Take a look at all the differences between the Embedding version of DotProductBias and the create_params version, and try to understand why each of those changes is required. If you're not sure, try reverting each change to see what happens. (NB: even the type of brackets used in forward has changed!)
---
NA

---
2. Find three other areas where collaborative filtering is being used, and find out what the pros and cons of this approach are in those areas.
---
NA

---
2. Complete this notebook using the full MovieLens dataset, and compare your results to online benchmarks. See if you can improve your accuracy. Look on the book's website and the fast.ai forum for ideas. Note that there are more columns in the full dataset—see if you can use those too (the next chapter might give you ideas).
---
NA

---
4. Create a model for MovieLens that works with cross-entropy loss, and compare it to the model in this chapter.
---
NA
