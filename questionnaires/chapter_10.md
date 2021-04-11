# Questionnaire: NLP Deep Dive: RNNs

---
1. What is "self-supervised learning"?
---
It is training a model using as labels some transformation of the input data eg. predicting a missing word in a text, placing frames of a video in the right order.

---
2. What is a "language model"?
---
It's a model trained usually using a large corpora to guess the next word or a missing one. The language model contains a lot of semantical and grammatical knowledge about the natural language in the training data as it needs all these information to do right predictions.

---
3. Why is a language model considered self-supervised?
---
It's self-supervised as no additional data or labels are needed to train a language model besides the text itself.

---
4. What are self-supervised models usually used for?
---
Self-supervised models are used normally for transfer learning as they embed useful information that is useful for downstream tasks requiring only a task specific fine tuning.

---
5. Why do we fine-tune language models?
---
To include domain specific vocabulary and knowledge e.g. a table in a StackOverflow dataset has a different meaning to the one in an e-commerce context.

---
6. What are the three steps to create a state-of-the-art text classifier?
---
- Pre-trained language model
- Domain specific language model
- Transfer learning ro a downstream task

---
7. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
---
The unlabeled movie reviews are used to fine-tune the language model before training the classifier with the labeled data.

---
8. What are the three steps to prepare your data for a language model?
---
- Text Tokenization
- Tokens Numericalization
- Data loader creation

---
9. What is "tokenization"? Why do we need it?
---
To make natural language understandable to a computer, we need to transform text into a valid format. Natural language is a discrete combinatorial system, so the first step is to split text into a finite set of words, substrings or characters known as `tokens`.

---
10.  Name three different approaches to tokenization.
---
- Word-based: split sentences by spaces and punctuation marks
- Subword based: split words into the most common substrings 
- Character-based: split text into individual characters

---
11. What is `xxbos`?
---
It's a special token in fastai (as some others that start with xx). It indicates the *beginning of stream* (BOS)

---
12.  List four rules that fastai applies to text during tokenization.
---

- **fix_html**: Replaces special HTML characters with a readable version 
- **replace_rep**: Replaces any character repeated three times or more with a special token for repetition (xxrep), the number of times it's repeated, then the character
- **rm_useless_spaces**: Removes all repetitions of the space character
- **replace_maj**: Lowercases a capitalized word and adds a special token for capitalized (xxmaj) in front of it
- **lowercase**: Lowercases all text and adds a special token at the beginning (xxbos) and/or the end (xxeos)

---
13.   Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
---
With this rule, the model needs only one token for the character keeping the information about the repetition so that the model can learn its meaning.

---
14.  What is "numericalization"?
---
It's the step where we convert the tokens to numbers as it's done normally with any categorical variable.

---
15. Why might there be words that are replaced with the "unknown word" token?
---
If the vocabulary is too large, it's a good idea to remove uncommon words from the vocabulary. Uncommon words could be typos or words in a different language that in most cases are not needed to create a useful language model.
Fastai has the parameter `max_vocab` in the [Numericalize](https://docs.fast.ai/text.data.html#Numericalize) class that we can use to control the size of our embeddings, words not in the vocabulary are labelled as `xxunk`.

---
16. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
---
The first row of the second batch contains the next 64 tokens while the
second row of the first batch will have the 64 tokens following the first row of the batch completing the first full sequence.

---
17.  Why do we need padding for text classification? Why don't we need it for language modeling?
---
For a language model, we are predicting the next word. In this case, a text can fallow the previous one as the model is learning from a sequence of text.
In classification, we need to keep every item in a row with its respective label. As each text has a different length, we need to pad the text so that all tensors have the same dimensions.

---
18. What does an embedding matrix for NLP contain? What is its shape?
---
It has a tensor of size n for every token (generally a word). Its shape is therefore (vocabulary_size, embeddings_dimensions)

---
19. What is "perplexity"?
---
Perplexity is the probability of predicting a sentence in the test set. The idea behind is that the power of the model to predict text not seen before, it's sign of the quality of the model.

$$pp = 2^{\frac{-1}{n}{\sum_{n=1}^{n}{log_2 LM(w_i|w_{1:i-1})}}}$$

---
20.   Why do we have to pass the vocabulary of the language model to the classifier data block?
---
To be sure the classifier has the same vocabulary index of the pre-trained embeddings.

---
21.  What is "gradual unfreezing"?
---
It has been shown that training pre-trained models gives better results if first the last layers are trained while keeping the previous ones frozen ina gradual manner.

---
22. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?
---
To train an automatic identification of machine-generated texts model, we need text generated from a machine as training data. Therefore, they are always a step behind as text generators models can always learn to create text that can't be easily detected by the latest model.

# Further Research

---
1. See what you can learn about language models and disinformation. What are the best language models today? Take a look at some of their outputs. Do you find them convincing? How could a bad actor best use such a model to create conflict and uncertainty?
---
NA

---
2. Given the limitation that models are unlikely to be able to consistently recognize machine-generated texts, what other approaches may be needed to handle large-scale disinformation campaigns that leverage deep learning?
---
NA
