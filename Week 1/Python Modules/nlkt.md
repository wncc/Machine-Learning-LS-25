# NLTK
NLTK is a leading Python library for working with human language data (text). It provides easy-to-use interfaces for over 50 corpora and lexical resources, and powerful tools for text processing.
## Features of NLTK
## 1. Tokenization
Tokenization is the process of splitting text into smaller units such as words or sentences.
Types of Tokenization in NLTK:
### 1. Word Tokenization
Splits text into individual words and punctuation.

```
import nltk
from nltk.tokenize import word_tokenize

text = "Hello, how are you?"
tokens = word_tokenize(text)
print(tokens)
```
### Output:
 ['Hello', ',', 'how', 'are', 'you', '?']

⚠️ Requires: nltk.download('punkt')

### 2. Sentence Tokenization
Splits text into sentences.

```
from nltk.tokenize import sent_tokenize

text = "Hello there. How are you doing? I hope you're well."
sentences = sent_tokenize(text)
print(sentences)
```
### Output: 
['Hello there.', 'How are you doing?', "I hope you're well."]

### 3. Regexp Tokenizer
Use regular expressions to define your own rules.
```
from nltk.tokenize import regexp_tokenize

text = "Email me at test@example.com"
tokens = regexp_tokenize(text, pattern=r'\S+')
print(tokens)
```
### Output:
['Email', 'me', 'at', 'test@example.com']

## 2. Stop-Words
Stopwords are common words that carry little meaningful information, such as:
"is", "the", "and", "in", "to" etc.
They're often removed in text preprocessing.

 ### 1. Import and Download Stopwords
```
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```
### 2. Get Stopwords List (e.g., English)
```
stop_words = set(stopwords.words('english'))
print(stop_words)
```
### 3. Remove Stopwords from a Text
```
from nltk.tokenize import word_tokenize
nltk.download('punkt')

text = "This is an example showing off stop word filtering."
words = word_tokenize(text)
filtered = [word for word in words if word.lower() not in stop_words]

print(filtered)
```
### Output:
```
['example', 'showing', 'stop', 'word', 'filtering', '.']
```

## Stemming
Stemming is the process of reducing a word to its base or root form (stem), usually by chopping off suffixes.

### PorterStemmer
```
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["playing", "played", "player", "plays"]

stems = [stemmer.stem(word) for word in words]
print(stems)
```
### Output:

['play', 'play', 'player', 'play']

### SnowballStemmer
Better and supports many language
```
from nltk.stem import SnowballStemmer

snowball = SnowballStemmer("english")
print(snowball.stem("running"))  # → 'run'
```
### Output:
run

### Stemming with Regex
Normally, stemming is done using NLP algorithms (like PorterStemmer). But you can mimic a very simple stemmer using regex by removing common suffixes (like -ing, -ed, -s) from words.

This is not true stemming, but it's a rough shortcut.
```
from nltk.stem import RegexpStemmer

# Strip common suffixes like -ing, -ed, -s
stemmer = RegexpStemmer(r'(ing|ed|s)$')

words = ['playing', 'played', 'plays', 'player', 'jumps']
stems = [stemmer.stem(word) for word in words]

print(stems)

```
### Output:
['play', 'play', 'play', 'player', 'jump']

## Lemmatizer
Lemmatization reduces a word to its base form (lemma), but uses vocabulary and grammar rules, so the result is a real word.

```
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running"))    # → 'running' (default POS is noun)
print(lemmatizer.lemmatize("running", pos="v"))  # → 'run'
print(lemmatizer.lemmatize("better", pos="a"))   # → 'good'
```
Lemmatizer needs to know if it's a verb, noun, adjective, etc.
Otherwise, it assumes it's a noun by default. That is why it is neccesary to mention the POS.
| POS       | Code  |
| --------- | ----- |
| Verb      | `"v"` |
| Noun      | `"n"` |
| Adjective | `"a"` |
| Adverb    | `"r"` |

## POS Tagging
Part-of-Speech (POS) tagging assigns a grammatical label (noun, verb, adjective, etc.) to each word in a sentence.

```
import nltk
from nltk import word_tokenize, pos_tag

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print(tags)
```
### Output:
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'),
 ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

 ### Common POS Tags:

| Tag   | Meaning          | Example    |
| ----- | ---------------- | ---------- |
| `NN`  | Noun (singular)  | dog, ball  |
| `VB`  | Verb (base)      | run, go    |
| `JJ`  | Adjective        | blue, lazy |
| `RB`  | Adverb           | quickly    |
| `DT`  | Determiner       | the, a     |
| `VBG` | Verb (gerund)    | running    |
| `VBZ` | Verb (3rd pers.) | eats       |
| `PRP` | Pronoun          | she, he    |

## Named Entity Recognisation
### What is Named Entity Recognition (NER)?
NER is about finding real-world entities in text, like:

- Person names

- Locations

- Organizations

- Dates

- Monetary values

```
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download models (only once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

text = "Barack Obama was born in Hawaii and became the President of the United States."

# Step 1: Tokenize
tokens = word_tokenize(text)

# Step 2: POS tagging
pos_tags = pos_tag(tokens)

# Step 3: Named Entity Chunking
named_entities = ne_chunk(pos_tags)

print(named_entities)
```
### Output
```
(S
  (PERSON Barack/NNP Obama/NNP)
  was/VBD
  born/VBN
  in/IN
  (GPE Hawaii/NNP)
  and/CC
  became/VBD
  the/DT
  (ORGANIZATION President/NNP)
  of/IN
  the/DT
  (GPE United/NNP States/NNPS)
  ./.)
  ```
  ### Drawing A graph
```
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Text input
text = "Elon Musk founded SpaceX in California."

# Tokenize and POS tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

# Perform Named Entity Recognition
named_entities = ne_chunk(pos_tags)

# Show GUI tree
named_entities.draw()
```

`Run This in Your PC to See A Interesting Graph`

## Additional Resources
[NLTK Tutorial](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/)
<br>







