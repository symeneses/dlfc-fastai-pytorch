# Questionnaire: Data Munging with fastai’s Mid-Level API

---
1. Why do we say that fastai has a "layered" API? What does it mean?
---
Because it has different layers of abstraction, increasing the complexity in each layer.
We have seen so fat the `Applications` layer (`Vision`, `Text`, `Tabular` and `Collab`) and the `High level` API (`Learner` and `DataBlock`).

---
2. Why does a `Transform` have a *decode* method? What does it do?
---
The decode method is used to reverse the transformation, so it is applied to the output, it will return the original input.

---
3. Why does a `Transform` have a *setup* method? What does it do?
---
The setup method initializes the inner state of the transform. After that, we can perform the respective transformation.

---
4. How does a `Transform` work when called on a tuple?
---
It will apply the same transformation to every item.

---
5. Which methods do you need to implement when writing your own `Transform`?
---
A `Transform` can be created using a decorator before the transformation function. If we need a `setup` or `decode` method, then we need to create a child class and implement the `setups` and `decodes` methods respectively.

---
6. Write a Normalize transform that fully normalizes items (subtract the mean and divide by the standard deviation of the dataset), and that can decode that behavior. Try not to peek!
---
```py
from fastai.data.all import Transform

class Normalize(Transform):
    def setups(self, items): 
        self.mean = sum(items)/len(items)
        self.sd = (sum([(i - self.mean)**2 for i in items])/len(items))**0.5
    def encodes(self, x): return (x - self.mean)/(self.sd) if self.sd != 0 else 1 
    def decodes(self, x): return x*self.sd + self.mean

tfm = Normalize()
tfm.setup([1, 2, 3, 4, 5])
start = 3
y = tfm(start)
z = tfm.decode(y)
f"mean={tfm.mean}, sd={tfm.sd}, encode={y}, decode={z}"
```
`mean=3.0, sd=1.4142135623730951, encode=0.0, decode=3.0`

---
7. Write a Transform that does the numericalization of tokenized texts (it should set its vocab automatically from the dataset seen and have a decode method). Look at the source code of fastai if you need help.
---
```py
from fastai.data.all import Transform

class Numericalize(Transform):
    def setups(self, tokens): 
        self.size, self.vocab = self._create_vocab(tokens)
        self.index = {v:k for k, v in self.vocab.items()}
    def encodes(self, x): return self.index[x.lower()] if x in self.index else 0
    def decodes(self, x): return self.vocab[x]

    @classmethod
    def _create_vocab(cls, tokens):
        vocab = {0: "unk"}
        i = 1
        for t in tokens:
            if t not in vocab.values():
                vocab[i] = t
                i += 1
        return i, vocab

tfm = Numericalize()
tfm.setup(['i', 'like', 'you', 'eat', 'a', 'lot', 'me'])
x = ("i", "like", "you", "a", "lot")
y = tfm(x)
z = tfm.decode(y)
f"vocab size={tfm.size}, sentence={' '.join([tfm.vocab[o] for o in y])}, encode={y}, decode={z}"
```
`vocab size=8, sentence=i like you a lot, encode=(1, 2, 3, 5, 6), decode=('i', 'like', 'you', 'a', 'lot')`

---
8. What is a `Pipeline`?
---
It's class in `fastai` the allows us to create sequences of transformations. This helps to understand better the code and is also really helpful to deploy systems on production where the same transformations applied for training are needed in inferencing time.

---
9.  What is a `TfmdLists`?
---
A `TfmdLists` is a class to apply a list of transformations or `Pipeline` to some input data, which can be split. 

---
10. What is a `Datasets`? How is it different from a `TfmdLists`?
---
`Datasets` can apply one or more pipelines to the same input object. This is util to create features and labels. 

---
11. Why are `TfmdLists` and `Datasets` named with an "s"?
---
Because they have a parameter `splits` with which we can split the input data in multiple objects.

---
12. How can you build a `DataLoaders` from a `TfmdLists` or a `Datasets`?
---
With the `dataloaders` method which has different customization options: after_item, before_match, after_batch.

---
13. How do you pass `item_tfms` and `batch_tfms` when building a `DataLoaders` from a `TfmdLists` or a `Datasets`?
---
 For `item_tfms`  with the parameter `after_item` and for `batch_tfms` with `after_batch`.

---
14. What do you need to do when you want to have your custom items work with methods like `show_batch` or `show_results`?
---
Implement the method `decode` in the respective `Transform`.

---
15. Why can we easily apply fastai data augmentation transforms to the `SiamesePair` we built?
---
Because transforms will apply to each image as the class `SiameseImage` is a `Tuple`.

# Further Research

---
1. Use the mid-level API to prepare the data in DataLoaders on your own datasets. Try this with the Pet dataset and the Adult dataset from Chapter 1.
---
NA

---
2. Look at the Siamese tutorial in the fastai documentation to learn how to customize the behavior of `show_batch` and `show_results` for new type of items. Implement it in your own project.
---
NA
