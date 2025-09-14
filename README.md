# Natural Language Processing (NLP) with NLTK

## Overview
Natural Language Processing (NLP) is a subfield of computer science and AI focused on the interaction between computers and human languages. It involves tasks like speech recognition, natural language understanding, and generation.

This project demonstrates key NLP concepts and builds a text classification system using NLTK and scikit-learn.

---

## Key Concepts Covered

### Tokenization
Breaking text into sentences or words using `sent_tokenize` and `word_tokenize`.

### Stop Words Removal
Filtering common words using NLTK's predefined stop words list.

### POS Tagging
Part-of-Speech tagging helps identify nouns, verbs, adjectives, etc., using `pos_tag`.

### Stemming & Lemmatization
- **Stemming** reduces words to their base form (e.g., "playing" → "play").
- **Lemmatization** reduces words to dictionary form considering context.

---

## Dataset Used
- NLTK movie_reviews corpus with 2000 labeled movie reviews (positive and negative).
- Text data preprocessed by removing stop words and punctuation.

---

## Feature Extraction
- Extracted top 3000 frequent words from training data.
- Each document represented as a feature dictionary with word counts.

---

## Classification Models

### SVM Classifier
Used `SklearnClassifier(SVC())` for text classification.

```python
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
classifier = SklearnClassifier(SVC())
classifier.train(training_data)
accuracy = nltk.classify.accuracy(classifier, testing_data)
print("SVM Accuracy:", accuracy)
```

Multinomial Naive Bayes Classifier

Used MultinomialNB() for text classification.

from sklearn.naive_bayes import MultinomialNB
classifier = SklearnClassifier(MultinomialNB())
classifier.train(training_data)
accuracy = nltk.classify.accuracy(classifier, testing_data)
print("Naive Bayes Accuracy:", accuracy)
```

Results

Both SVM and Naive Bayes achieved high accuracy on the text classification task. This project demonstrates the power of NLP preprocessing and classic machine learning models in handling text data.

References

NLTK Documentation

Stanford NLP Book
