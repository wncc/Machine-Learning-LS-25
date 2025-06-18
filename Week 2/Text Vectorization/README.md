# Text Vectorization: TF-IDF vs. Bag-of-Words

We'll explore the concepts of **Bag-of-Words** and **TF-IDF**, which are fundamental techniques for converting text into a numerical format that machine learning models can understand.

## Learning Resources

* **[Bag of Words (BoW) in NLP](https://youtu.be/pF9wCgUbRtc?si=K2lu22g747YvIeb2)**

    Overview of the Bag-of-Words technique, explaining how it converts text into numerical representations by counting word frequencies. It also discusses the pros and cons of this method, such as its simplicity versus its inability to capture context and word order.

* **[TF-IDF (Term Frequency-Inverse Document Frequency)](https://youtu.be/OymqCnh-APA?si=vWxnEpyoGJThgSUU)**
    
    Introduction to TF-IDF, a more advanced metric that determines the importance of a word in a document relative to a collection of documents. It breaks down the concepts of Term Frequency (TF) and Inverse Document Frequency (IDF).

## Reading Material

For a deeper dive into the implementation of these concepts using Python's `scikit-learn` library, please refer to the following articles:

* **[Using CountVectorizer to Extracting Features from Text](https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/)**
    
    This article explains how to use `CountVectorizer` from the `scikit-learn` library to implement the Bag-of-Words model.

* **[Understanding TF-IDF (Term Frequency-Inverse Document Frequency)](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/)**
    
    This article provides a detailed explanation of the TF-IDF statistic and its components.

## Comparison: Bag-of-Words vs. TF-IDF

| Feature                | Bag-of-Words (BoW)                                                  | Term Frequency-Inverse Document Frequency (TF-IDF)                                 |
| :--------------------- | :------------------------------------------------------------------ | :--------------------------------------------------------------------------------- |
| **Core Idea** | Counts the frequency of each word in a document.                    | Weighs words based on their frequency in a document and their rarity across all documents. |
| **Word Importance** | All words are treated equally. Common words can dominate.           | Gives higher weight to words that are frequent in a document but rare in the corpus. |
| **Context** | Does not capture the context or semantic meaning of words.          | Also does not capture context, but it can better identify important, topic-specific words. |
| **Vector Representation** | Vectors contain raw word counts or frequencies.                     | Vectors contain weighted scores ($TF \times IDF$ values) for each word.                      |
| **Use Case** | Good for simple text classification tasks where word counts are sufficient. | Better for more complex tasks like search engines and document clustering.        |
