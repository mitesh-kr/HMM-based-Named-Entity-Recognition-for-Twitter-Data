# Named Entity Recognition using Hidden Markov Models for Twitter Data

## Abstract
This report presents a Hidden Markov Model (HMM) based approach for Named Entity Recognition (NER) on Twitter data. We implement and evaluate different configurations of HMM models, including bigram and trigram models with and without contextual emission probabilities. Our results demonstrate the effectiveness of higher-order models and context-aware emission probabilities for improving NER performance on social media text.

## 1. Introduction

Named Entity Recognition (NER) is a fundamental task in natural language processing that involves identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, etc. Twitter data presents unique challenges for NER due to its informal nature, non-standard orthography, and limited context.

In this project, we implement HMM-based NER models to identify 10 types of named entities in Twitter data:
1. Person
2. Product
3. Company
4. Geolocation
5. Movie
6. Music artist
7. TVShow
8. Facility
9. Sports team
10. Others

## 2. Background

### 2.1 Hidden Markov Models

A Hidden Markov Model (HMM) is a statistical model where the system being modeled is assumed to be a Markov process with hidden states. In the context of NER, words are the observable outputs, and the named entity tags are the hidden states we want to infer.

The HMM is characterized by:
- A set of states (NER tags)
- Start probabilities (π)
- Transition probabilities (A) between states
- Emission probabilities (B) from states to observations

### 2.2 Parameter Estimation

The HMM parameters are estimated from annotated training data:
1. **Start probabilities (π)**: The probability of a sequence starting with a particular tag
2. **Transition probabilities (A)**: The probability of transitioning from one tag to another
3. **Emission probabilities (B)**: The probability of observing a word given a specific tag

### 2.3 Viterbi Algorithm

The Viterbi algorithm is used for decoding, i.e., finding the most likely sequence of tags given a sequence of words:
- For each position and possible tag, it computes the probability of the most likely path ending with that tag
- It uses dynamic programming to efficiently search through the exponential space of possible tag sequences

## 3. Methodology

### 3.1 Data Preprocessing

The Twitter NER dataset consists of words annotated with their corresponding named entity tags. We preprocess the data by:
- Tokenizing each sentence
- Converting to lowercase (optional)
- Handling special characters and emojis
- Managing rare words and out-of-vocabulary terms

### 3.2 Model Variants

We implement four variants of HMM models:

1. **Bigram model without context in emission probabilities**:
   - Uses first-order transitions (tag_i depends only on tag_{i-1})
   - Emission probabilities consider only the current tag

2. **Bigram model with context in emission probabilities**:
   - Uses first-order transitions
   - Emission probabilities consider both the current tag and previous tag

3. **Trigram model without context in emission probabilities**:
   - Uses second-order transitions (tag_i depends on tag_{i-1} and tag_{i-2})
   - Emission probabilities consider only the current tag

4. **Trigram model with context in emission probabilities**:
   - Uses second-order transitions
   - Emission probabilities consider the current tag, previous tag, and tag before the previous

### 3.3 Parameter Estimation

For each model variant, we estimate:

1. **Start probabilities (π)**:
   ```
   π(tag) = Count(sentences starting with tag) / Total number of sentences
   ```

2. **Transition probabilities (A)**:
   - For bigram models:
     ```
     A(tag_j|tag_i) = Count(tag_i followed by tag_j) / Count(tag_i)
     ```
   - For trigram models:
     ```
     A(tag_k|tag_i,tag_j) = Count(tag_i, tag_j followed by tag_k) / Count(tag_i, tag_j)
     ```

3. **Emission probabilities (B)**:
   - Without context:
     ```
     B(word|tag) = Count(word with tag) / Count(tag)
     ```
   - With context (bigram model):
     ```
     B(word|tag, prev_tag) = Count(word with tag and previous tag = prev_tag) / Count(tag, prev_tag)
     ```
   - With context (trigram model):
     ```
     B(word|tag, prev_tag, prev_prev_tag) = Count(word with tag, prev_tag, prev_prev_tag) / Count(tag, prev_tag, prev_prev_tag)
     ```

### 3.4 Smoothing

To handle sparse data and unseen transitions/emissions, we apply smoothing techniques:
- Laplace (add-one) smoothing for transition probabilities
- Good-Turing discounting for emission probabilities
- Backoff model for unseen emissions

### 3.5 Decoding

We use the Viterbi algorithm to find the most likely sequence of tags given a sequence of words:
1. Initialize the trellis with starting probabilities
2. For each word, compute the most likely path to each possible tag
3. Backtrack to find the optimal tag sequence

## 4. Experimental Setup

### 4.1 Dataset

The dataset consists of Twitter posts annotated with named entity tags:
- Training set: [Size of training set] sentences
- Validation set: [Size of validation set] sentences
- Test set: [Size of test set] sentences

### 4.2 Evaluation Metrics

We evaluate our models using:
- **Accuracy**: The proportion of correctly predicted tags
- **Precision**: The proportion of predicted entities that are correct
- **Recall**: The proportion of actual entities that are correctly predicted
- **F1-score**: The harmonic mean of precision and recall

### 4.3 Implementation

The models are implemented in Python with the following structure:
- `hmm_tagger.py`: Core HMM implementation
- `data_utils.py`: Data loading and preprocessing
- `evaluation.py`: Metrics computation
- `train.py`: Model training script
- `predict.py`: Prediction script
- `evaluate.py`: Evaluation script

## 5. Results and Discussion


| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Bigram (No Context) | 0.6378 | 0.8949 | 0.6378 | 0.7424 |
| Bigram (With Context) | 0.6748 | 0.8746 | 0.6748 | 0.7597 |
| Trigram (No Context) | 0.9029 | 0.8168 | 0.9029 | 0.8577 |
| Trigram (With Context) | 0.9033 | 0.8168 | 0.9033 | 0.8579 |




## 6. Conclusions

[Summary of key findings]

1. HMM-based approaches can achieve reasonable performance for NER on Twitter data,
with the best model achieving an F1 score of 0.76.

2. Adding context to emission probabilities consistently improves performance, highlighting
the importance of considering surrounding tokens.

3. The extreme class imbalance in the dataset poses significant challenges, particularly for
more complex models like the trigram HMM.

4. Simpler models (bigram) outperform more complex ones (trigram) in this task due to
their better handling of data sparsity and class imbalance.

Overall, while the HMM approach provides a strong baseline for NER on Twitter data, the task
remains challenging due to the nature of social media text and the imbalanced distribution of
entity types. Future work should focus on addressing these specific challenges to improve
performance further.


## 7. References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
2. Ritter, A., Clark, S., Mausam, & Etzioni, O. (2011). Named entity recognition in tweets: An experimental study. EMNLP.
3. Lafferty, J., McCallum, A., & Pereira, F. C. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. ICML.
4. Jurafsky, D., & Martin, J. H. (2009). Speech and language processing. Pearson Education India.
5. [Additional relevant references]
