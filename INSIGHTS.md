# Insights

I learned several important concepts and techniques in this project:

### Model Persistence
- **Three types of model saving**: 
  - `pickle` for general Python objects
  - `joblib` for scikit-learn models (more efficient for NumPy arrays)
  - Framework-specific methods like `model.save()` for Keras/TensorFlow models and `save_pretrained()` for PyTorch transformers

### Hyperparameter Tuning
- **Keras Tuner**: Automated hyperparameter search for deep learning models
- **GridSearchCV**: Exhaustive hyperparameter search for scikit-learn models with cross-validation

### Progress Tracking
- **TQDM**: Progress bars for monitoring long-running operations and training loops

### Data Management
- **Train-validation-test split**: Importance of proper data distribution for unbiased model evaluation
- **Class distribution**: Understanding class imbalance and its impact on model performance, including the use of balanced class weights

### ML Pipelines
- **Scikit-learn Pipeline**: Combining preprocessing steps (vectorization) and model training into a single pipeline for consistency and reproducibility

### Text Preprocessing and Vectorization
- **Tokenization approaches**:
  - Custom tokenizers (Keras Tokenizer)
  - TextVectorization layer (TensorFlow)
  - BertTokenizer (Hugging Face transformers)
- **Vectorization methods**:
  - CountVectorizer (bag-of-words)
  - TF-IDF Vectorizer (term frequency-inverse document frequency)

### Deep Learning (RNN Model Architecture)
- **Input Layer**: Accepts raw text input
- **TextVectorization Layer**: Converts text to integer sequences
- **Embedding Layer**: Maps tokens to dense vectors
- **Conv1D Layer**: Detects local patterns in sequences
- **MaxPooling1D Layer**: Down-samples sequences by keeping strongest features
- **LSTM Layer**: Captures long-term dependencies and sentence-level context
- **Dropout Layers**: Prevents overfitting by randomly dropping features
- **GlobalAveragePooling1D Layer**: Summarizes sequence into single vector
- **Dense Output Layer**: Final classification with softmax activation

### Transfer Learning
- **Fine-tuning LLMs**: Adapting pre-trained transformer models (BERT) for specific downstream tasks like sentiment analysis

### Framework Differences
- **PyTorch**: Flexible on-the-fly data transformation approach, allowing dynamic preprocessing during training
- **TensorFlow/Keras**: Pre-configured data transformation approach, where preprocessing is typically set up before training

### Data Leakage Prevention
- **Max length setting**: Critical importance of setting sequence length parameters (like `maxlen`) based only on training data statistics, not test data, to prevent data leakage and ensure realistic model evaluation