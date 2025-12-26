import re
import nltk
from nltk.stem.porter import PorterStemmer
import spacy

# download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# step 1
def to_lowercase(text):
    """
    Convert text to lowercase.
    
    Args:
        text (str): Input text to convert
        
    Returns:
        str: Text in lowercase
    """
    if text == '':
        return text
    return text.lower()

###########################################

# step 2
def remove_newlines(text):
    """
    Remove newline characters from text.
    
    Args:
        text (str): Input text that may contain newline characters
        
    Returns:
        str: Text with newline characters removed
    """
    if text == '':
        return text
    return text.replace('\n', ' ').replace('\r', ' ')

###########################################

# step 3
def remove_punctuation(text):
    """
    Remove punctuation from text.
    
    Args:
        text (str): Input text that may contain punctuation
        
    Returns:
        str: Text with punctuation removed
    """
    if text == '':
        return text
    # remove all non-alphanumeric, non-whitespace characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

###########################################

# step 4
def remove_stopwords(text):
    """
    Remove stop words from text, except 'not' and other negation words.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stop words removed (except 'not')
    """
    if text == '':
        return text
    
    # get English stopwords and remove 'not' from the stopwords set
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.discard('not')
    
    # split text into words
    words = text.split()
    
    # filter out stopwords
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    
    # join words back into text
    return ' '.join(filtered_words)

###########################################

# step 5
def apply_stemming(text):
    """
    Apply Porter Stemmer to each word in the text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with words stemmed
    """
    if text == '':
        return text
    
    ps = PorterStemmer()
    
    # split text into words
    words = text.split()
    
    # apply stemming to each word
    stemmed_words = [ps.stem(word) for word in words]
    
    # join words back into text
    return ' '.join(stemmed_words)


###########################################

# step 6
def remove_digits(text):
    """
    Remove digits from text unless they are related to sentiment 
    (e.g., "10 out of 10", "5 stars", "rating 4").
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with non-sentiment-related digits removed
    """
    if text == '':
        return text
    
    # keywords that indicate sentiment-related numbers
    sentiment_keywords = ['out of', 'stars', 'star', 'rating', 'ratings', 
                         'score', 'scores', 'mark', 'marks', 'points', 'point']
    
    # pattern to find numbers (including standalone digits and numbers in context)
    def should_keep_digit(match):
        # get the matched number and its position
        start_pos = match.start()
        end_pos = match.end()
        
        # get context around the number (20 characters before and after)
        context_start = max(0, start_pos - 20)
        context_end = min(len(text), end_pos + 20)
        context = text[context_start:context_end].lower()
        
        # check if any sentiment keyword is in the context
        for keyword in sentiment_keywords:
            if keyword in context:
                return True
        
        # if no sentiment keyword found, remove the number
        return False
    
    # replace numbers: keep if sentiment-related, remove otherwise
    ### \b represents a word boundary to ensure that the match is a separate word (not part of another word).
    ### \d+ matches one or more digits.
    ### r'\b\d+\b' represents "123" in "hello 123 hello", not in "hello12 3hello"
    ### group() extracts the part of the string that matched the regular expression.
    ### re.sub(pattern, replacement, text) searches for occurrences of the pattern in the text and replaces them with the replacement.
    result = re.sub(r'\b\d+\b', lambda m: m.group() if should_keep_digit(m) else '', text)
    
    # clean up multiple spaces that might be created
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

###########################################

# step 7
def apply_lemmatization(text, nlp):
    """
    Apply lemmatization using spaCy to the text.
    
    Args:
        text (str): Input text
        nlp: spaCy pre-trained language model
        
    Returns:
        str: Text with words lemmatized
    """
    if text == '':
        return text
        
    # process text with spaCy
    doc = nlp(text)
    
    # extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    # join tokens back into text
    return ' '.join(lemmatized_tokens)

###########################################

# MASTER FUNCTION that applies all preprocessing steps
def preprocess_text(text, nlp=spacy.load("en_core_web_sm")):
    """
    Apply all preprocessing steps to a text.
    
    Args:
        text (str): Input text
        nlp: spaCy pre-trained language model
        
    Returns:
        str: Fully preprocessed text
    """
    text = to_lowercase(text)
    text = remove_newlines(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    text = remove_digits(text)
    text = apply_lemmatization(text, nlp)
    return text