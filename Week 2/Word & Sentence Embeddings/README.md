# What are Embeddings?
Embeddings are vector representations of text (words, phrases, or sentences) that capture semantic meaning in a way that machines can understand. Instead of working with raw text, we transform it into dense vectors (usually floating-point numbers) that encode similarity and relationships.
## Word Embeddings

Word embeddings represent individual words as vectors. Words with similar meanings are mapped to nearby points in vector space.

### Common Word Embedding Methods:
- ### Word2Vec (by Google):

  Learns word vectors using context (CBOW and Skip-gram models).

  Example: king - man + woman ≈ queen

- ### GloVe (by Stanford):

  Uses word co-occurrence counts in a large corpus.

  Captures both local and global word statistics.

- ### FastText (by Facebook):
  Builds word vectors using subword (character n-gram) information.

  Helps with rare and out-of-vocabulary (OOV) words.

### Pros:
Captures semantic and syntactic relations.

Works well for many NLP tasks like sentiment analysis, NER.

### Limitations:
Fixed meaning for each word (e.g., "bank" of river vs. bank as financial institution).

Doesn’t consider word order or context

## Sentence Embeddings
Sentence embeddings represent entire sentences (or even paragraphs) as single vectors.

These embeddings take into account:

Word order

Context

Syntax & semantics

### Common Sentence Embedding Methods:
- ### Universal Sentence Encoder (USE):

  Pretrained model by Google.

  Converts sentences to 512-dimensional vectors.

- ### Sentence-BERT (SBERT):

  A modification of BERT to produce meaningful sentence-level embeddings.

  Good for tasks like sentence similarity, semantic search.

- ### InferSent:

  Trained on natural language inference data to encode sentences with general-purpose meaning.

### Pros:
Encodes full sentence meaning and context.

Useful for tasks like semantic similarity, document classification, retrieval, etc.

### Limitations:
Computationally heavier than word embeddings.

May require fine-tuning for domain-specific tasks


This week we will focus on Word2Vec, avgWord2vec and BERT

## Word2Vec (Word-Enbedding)
Word2Vec is a popular word embedding technique developed by Google in 2013. It represents words as dense vectors in a continuous vector space where semantic similarity between words is captured using their context in a large text corpus.

Word2Vec uses a shallow neural network and comes in two main variants:

- CBOW (Continuous Bag of Words): Predicts the current word from surrounding context.

- Skip-gram: Predicts surrounding context words given the current word.

Word2Vec learns how words relate to each other by analyzing how often they appear together in text. It then represents each word as a dense vector of numbers (typically 100–300 dimensions), where:

Similar words have similar vectors

Words used in similar contexts get mapped close together in space

The model we are using for implementation, is trained over 3-billion words and gives a 300 items array for each word

### Using Word2Vec with Gensim & Google’s Pretrained Model
Step 1: Install Gensim
``` 
pip install gensim
```
Step 2: Loading The Dataset

```
import gensim.downloader as api

model = api.load('word2vec-google-news-300') 
# downloads ~1.6GB model
```
Checking if model works :
```
print(model['king'])  # see vector
print(model.most_similar('king'))  # analogies
```
### More functions
```
from gensim.models import KeyedVectors

# Load the pre-trained model (you’ve already done this probably)
model_path = "/content/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# 1. Most similar words
print("📍 Most similar to 'dog':")
print(model.most_similar('dog'))

# 2. Word analogy: king - man + woman = ?
print("\n👑 Word analogy: king - man + woman = ?")
print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))

# 3. Word similarity
print("\n📏 Similarity between 'coffee' and 'tea':")
print(model.similarity('coffee', 'tea'))

# 4. Check if a word exists
print("\n🔎 Is 'dragon' in vocabulary?")
print('dragon' in model.key_to_index)
```
## Avg Word2Vec (Sentence Embedding)
Avg Word2Vec is a simple and effective method to convert an entire sentence or document into a single fixed-size vector, using pre-trained word vectors like Word2Vec.

### How Does It Work?
- Split the sentence into words.

- Look up the Word2Vec vector for each word.

- Average all the vectors (i.e., element-wise mean).

```
import numpy as np
from gensim.models import KeyedVectors

# Load Google News vectors
model_path = "/content/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def avg_word2vec(sentence, model):
    words = sentence.lower().split()
    valid_vectors = [model[word] for word in words if word in model]
    
    if not valid_vectors:
        return np.zeros(model.vector_size)
    
    return np.mean(valid_vectors, axis=0)

# Try it!
sentence = "I love machine learning"
vector = avg_word2vec(sentence, model)

print("🔢 Sentence Vector (shape):", vector.shape)
print("📈 First 5 dimensions:", vector[:5])
```

### Why Use Avg Word2Vec?
- Super simple to implement

- Fast and efficient

- Doesn’t capture word order or contextual meaning

- Best for: baseline models, quick semantic similarity, etc.

### BERT (Sentence Embedding)
BERT is a transformer-based model developed by Google AI in 2018. It’s trained to understand the context of a word in a sentence in both directions — left and right — making it bidirectional.

### Pretraining Tasks
- ` Masked Language Modeling (MLM)`:

  Random words are masked → BERT learns to predict them

- ` Next Sentence Prediction (NSP)`:

  Learns sentence relationships


### Limitations
- Large and slow

- Not ideal for sentence-level tasks → use SBERT instead

- Replaced by newer models like RoBERTa, DistilBERT, DeBERTa, GPT, etc

We will learn more about transformer artitecture and more detail of these model in the next week

[Resources](https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/)<br>
[Video Resource 1](https://youtu.be/hQwFeIupNP0?si=HDves-sB-DIqKPZf)<br>
[Video Resource 2](https://youtu.be/cyvkMFZnheo?si=OWsqEUaLE-ffqASl) 