## Assignment 2.1: Text Vectorization Implementation

### Objective

Manually implement the TF-IDF algorithm and compare the results with the outputs from scikit-learn's `CountVectorizer` and `TfidfVectorizer`.

### Corpus
```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
```

### Requirements & Guidelines
* You may use any standard Python data structures (e.g., lists, dictionaries).
* Use Python's `math` library for any necessary logarithmic calculations.
* Submit your Jupyter Notebook (`.ipynb`) or Python script (`.py`), along with a `Readme.md` file that summarizes your findings. Your summary should address:
    * A comparison of word scores between your manual TF-IDF, `CountVectorizer`, and `TfidfVectorizer`.
    * An explanation for why the scores for common words (like 'the') differ significantly between the methods.
