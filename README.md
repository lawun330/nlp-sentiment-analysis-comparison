# Sentiment Analysis: A Self-Study in NLP Model Comparison

This is a self-study project focused on Natural Language Processing (NLP) and various machine learning models for sentiment analysis. The project explores different model architectures and techniques to classify customer feedback as positive, negative, or neutral sentiment.

A key challenge addressed in this project is the mismatch between training and testing data: models are trained on relatively short sentences (average 3-4 words) but must predict sentiment for much longer sentences (average 65 words).

This length discrepancy motivated the exploration of aggregated prediction techniques, where long sentences are split into smaller chunks, predicted separately, and their probabilities are combined to make a final prediction.

## Customized Preprocessing

Several preprocessing techniques are used to improve the performance of NLP models and ensure the data is clean and suitable for training:

- Removing unwanted characters and special symbols
- Eliminating stop words (commonly used words like "and," "the," etc.)
- Applying stemming and lemmatization to reduce words to their root forms
- Encoding textual data
- Padding sequences to ensure uniform input lengths (required only for sequence-based models; not needed for bag-of-words models)

The preprocessing pipeline is implemented in `scripts/word_normalization.py` for consistent text transformation across all models.

## Project Structure

```
root/
├── datasets/           # Dataset files (raw and processed)
├── models/             # Trained model files (.joblib, .keras, PyTorch models)
├── notebooks/          # Jupyter notebooks for model development and analysis
├── scripts/            # Utility scripts (preprocessing, aggregated prediction)
├── text_transformers/  # Saved text vectorization objects (.pkl files)
└── tuner_results/      # Hyperparameter tuning results and trial data
```

## Models Explored

The project compares six different model architectures:

1. **Naive Bayes Classifier (NBC)**: Traditional probabilistic classifier using count vectorization
2. **Recurrent Neural Network (RNN)**: Deep learning model with two variants:
   - RNN with custom tokenizer
   - RNN with TextVectorization layer
3. **BERT Transformer**: Pre-trained transformer model fine-tuned for sentiment analysis
4. **Logistic Regression**: Linear classifier with TF-IDF vectorization
5. **Support Vector Machine (SVM)**: Linear kernel SVM with TF-IDF vectorization

## Project Workflow

### Phase 1: Data Preparation
- Combined multiple datasets to create a diverse training set
- Applied customized preprocessing pipeline
- Split data into training and testing sets
- Saved processed datasets for consistent model evaluation

### Phase 2: Model Development
- Trained individual models with appropriate preprocessing
- Each model was optimized with hyperparameter tuning
- Models were saved for later comparison and evaluation

### Phase 3: Baseline Comparison
- Compared all six models on standardized test datasets
- Evaluated performance using confusion matrices and accuracy metrics
- Identified BERT as the best-performing model

### Phase 4: Aggregated Prediction Hypothesis
- Hypothesized that splitting long sentences into chunks might improve performance
- Implemented aggregated prediction approach for five models (excluding BERT)
- Tested chunk-based prediction with probability averaging
- Found that Logistic Regression benefited from aggregation, while others did not

### Phase 5: Final Comparison
- Compared BERT (normal prediction) against Logistic Regression (aggregated prediction)
- Validated that BERT remains the superior model even when compared to the best aggregated approach

## Dataset Sources
- [given dataset for training](https://drive.google.com/file/d/1_SHjQJVxZdr_LW2aIHAiOSBPWWGWd7Bs/view?usp=drive_link)
- [open source dataset for training](https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset)
- [open source dataset for testing](https://www.kaggle.com/datasets/ahmedabdulhamid/reviews-dataset)
- (optional) [open source dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Notes

This project serves as a learning exercise in NLP and machine learning, exploring various approaches to sentiment analysis. The notebooks document the complete workflow from data preparation to model comparison and evaluation.