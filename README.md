# Sentiment-Analysis-on-Sentiment140-and-RTMR
A machine learning-based sentiment analysis tool comparing Naive Bayes, CNN, and LSTM models on Rotten Tomatoes and Sentiment140 datasets.
# Sentiment Analysis Using Machine Learning Models

This project focuses on sentiment analysis using machine learning techniques applied to two datasets: Rotten Tomatoes Movie Reviews (RTMR) and Sentiment140 (S140). The project implements and compares the performance of three models: Naive Bayes, Convolutional Neural Networks (CNN), and Long Short-Term Memory Networks (LSTM).

## Features

- **Preprocessing:** Tokenization, removal of noise, and lemmatization.
- **Text Representation:** CountVectorizer for Naive Bayes and sequence embeddings for CNN and LSTM.
- **Models:** Comparison of Naive Bayes, CNN, and LSTM on sentiment prediction.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Data Preprocessing](#data-preprocessing)
4. [Text Representation](#text-representation)
5. [Machine Learning Models](#machine-learning-models)
6. [Experimental Results](#experimental-results)
7. [Discussion](#discussion)
8. [Usage](#usage)
9. [References](#references)

## Introduction

Sentiment analysis involves understanding the sentiment conveyed in a text. This project leverages Naive Bayes, CNN, and LSTM models to classify text into predefined sentiment categories.

## Datasets

- **RTMR:** Annotated movie reviews with sentiments (0 - Negative to 4 - Positive).
- **S140:** Tweets labeled as positive (4), neutral (2), or negative (0).

## Data Preprocessing

The preprocessing steps include:
1. Tokenization
2. Removing hashtags, URLs, and stopwords
3. Converting text to lowercase
4. Lemmatization and filtering non-alphanumeric tokens
5. Padding sequences for neural network models

## Text Representation

- **Naive Bayes:** Used `CountVectorizer` from scikit-learn for a matrix of token counts.
- **CNN/LSTM:** Used Keras' `texts_to_sequences()` for token indexing, followed by sequence padding.

## Machine Learning Models

### 1. Naive Bayes
A probabilistic classifier using token counts as features.

### 2. CNN
Architecture:
- Embedding layer
- Convolution and pooling layers
- Dropout and batch normalization
- Dense layer with softmax activation

### 3. LSTM
Architecture:
- Embedding layer
- LSTM layer for sequence learning
- Dropout and dense layers

## Experimental Results

Results are evaluated using accuracy, precision, recall, and F1-score. Cross-dataset testing reveals challenges due to label inconsistencies:
- RTMR models achieved higher accuracy on RTMR test data compared to S140 test data and vice versa.
- Naive Bayes generally performed well within the dataset but struggled across datasets.
- Neural networks (CNN/LSTM) had mixed results, with CNN slightly outperforming LSTM.

## Discussion

Key observations:
- The absence of intermediate labels in the S140 dataset (e.g., "somewhat positive") impacted cross-dataset performance.
- Future work could explore transfer learning and hyperparameter tuning to improve model generalization.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git

2. Install the dependencies:
   ```bash
3. Run the Jupyter notebook to train and test models:
   ```bash
   jupyter notebook Sentiment.ipynb

References
Feldman, R. "Techniques and Applications for Sentiment Analysis." Communications of the ACM, 2013.
O'Shea, K., & Nash, R. "An Introduction to Convolutional Neural Networks." arXiv preprint, 2015.
Yu, Y., et al. "A Review of Recurrent Neural Networks: LSTM Cells and Network Architectures." Neural Computation, 2019.
Ruder, S., et al. "Transfer Learning in Natural Language Processing." NAACL Tutorials, 2019.
