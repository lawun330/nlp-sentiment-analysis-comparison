import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# step 1: convert a long sentence into a combination of shorter phrases
def chunk_sentence(sentence, chunk_size=4):
    """
    Convert a long sentence into a combination of shorter phrases by splitting
    it into chunks of specified size.
    
    Args:
        sentence (str): Input sentence to be chunked
        chunk_size (int): Number of words per chunk (default: 4)
        
    Returns:
        list: List of phrase strings, where each phrase contains up to 
              chunk_size words. The last phrase may contain fewer words 
              if the sentence length is not divisible by chunk_size.
    """
    words = sentence.split()
    total_words = len(words)
    full_chunks_count = total_words // chunk_size
    full_chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, full_chunks_count * chunk_size, chunk_size)]
    remaining = [' '.join(words[full_chunks_count * chunk_size:])]
    all_phrases = full_chunks + (remaining if remaining else [])

    return all_phrases

###########################################

# step 2: get average probabilities of a split sentence
def get_avg_probs(phrases, model, model_name, total_classes=3, tokenizer=None, maxlen=80):
    """
    Get average probabilities across all phrases by predicting each phrase
    separately and computing the mean probability for each class.
    
    Args:
        phrases (list): List of phrase strings to predict
        model: Trained classifier model with predict_proba method
        model_name (str): Name of the model
        total_classes (int): Total number of classes (default: 3)
        tokenizer: Tokenizer object for the model
        maxlen: Maximum length of the sequences
        
    Returns:
        numpy.ndarray: Array of average probabilities for each class, 
                       computed by averaging probabilities across all phrases
    """
    all_probs = np.zeros((len(phrases), total_classes))

    for i, phrase in enumerate(phrases):
        if model_name=="nbc":
            y_pred_probs = model.predict_proba([phrase])
        elif model_name=="rnn":
            y_pred_probs = model.predict(tf.convert_to_tensor([phrase]))
        elif model_name=="rnn_tokenizer":
            sequences = tokenizer.texts_to_sequences([phrase])
            padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
            y_pred_probs = model.predict(padded_sequences)
        elif model_name=="logr":
            y_pred_probs = model.predict_proba([phrase])
        elif model_name=="svm":
            y_pred_probs = model.predict_proba([phrase])
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        all_probs[i] = y_pred_probs

    avg_probs = all_probs.mean(axis=0)
    
    return avg_probs

###########################################

# step 3: make aggregated prediction
def get_aggregated_prediction(sentence, model, model_name, chunk_size=4, total_classes=3, tokenizer=None, maxlen=80):
    """
    Make an aggregated prediction for a sentence by chunking it into phrases,
    predicting probabilities for each phrase, and selecting the class with
    the highest average probability.
    
    Args:
        sentence (str): Input sentence to predict
        model: Trained classifier model with predict_proba method
        model_name (str): Name of the model
        chunk_size (int): Number of words per chunk (default: 4)
        total_classes (int): Total number of classes (default: 3)
        tokenizer: Tokenizer object for the model
        maxlen: Maximum length of the sequences

    Returns:
        int: Predicted class index (the class with the highest average 
             probability across all phrase predictions)
    """
    phrases = chunk_sentence(sentence, chunk_size)
    avg_probs = get_avg_probs(phrases, model, model_name, total_classes, tokenizer, maxlen)
    final_class = int(avg_probs.argmax())

    return final_class