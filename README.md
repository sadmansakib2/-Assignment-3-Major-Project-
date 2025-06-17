
# Toxic Comment Classification using DistilBERT

## Problem Statement
Toxic comments on online platforms harm user experience and mental well-being. This project focuses on detecting toxic language in comments using natural language processing and fine-tuning a pre-trained model.

## Dataset
The dataset used is a subset of the Jigsaw Toxic Comment Classification dataset from Kaggle.
- Total samples used: 5,000
- Labels: 0 = Non-Toxic, 1 = Toxic
- Columns used: `comment_text` and `toxic`

## Model and Method
- Base model: `distilbert-base-uncased`
- Preprocessing: Tokenization using HuggingFace tokenizer
- Truncation and padding applied
- Fine-tuned using `Trainer` API from HuggingFace `transformers`
- Split: 80% training, 20% test

## Training Setup
- Epochs: 2
- Batch size: 16 (train), 32 (eval)
- Optimizer: AdamW (default)
- Learning rate: 2e-5

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix plotted for test performance
- Sample predictions reviewed manually

## Insights
- Token length analysis showed most comments are within safe limits for DistilBERT
- Minor class imbalance observed
- Model achieved strong performance metrics suitable for binary classification

## Challenges
- Class imbalance slightly impacted precision
- Training time on Colab was limited, so full dataset not used

## Future Work
- Apply weighted loss or oversampling for class imbalance
- Try RoBERTa or BERT-base for comparison
- Perform hyperparameter tuning for further improvements

---

Author: Sadman Sakib
Course: COMP8420
