import string
import re
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

def preprocess_document(document, lemmatizer=WordNetLemmatizer()):
    """Apply NLP preprocessing to a given document.

    Args:
        document (str): The document to be preprocessed.
        lemmatizer (WordNetLemmatizer): The lemmatizer to be used.
    """     
    # Convert to lower case
    document = document.lower()

    # Tokenize document
    document_tokens = word_tokenize(document)
    
    # Remove stopwords
    document_tokens = [word for word in document_tokens if word not in stopwords.words('english')]
    
    # Remove punctuation from tokens
    extended_punctuation = string.punctuation + '“”‘’—'
    document_tokens = [''.join(char for char in s if char not in extended_punctuation) for s in document_tokens]
    
    # Perform lemmatization
    document_tokens = [lemmatizer.lemmatize(word) for word in document_tokens]

    return document_tokens

def prepare_dataset(df):
    """Prepare the dataset for training a machine learning model.

    Args:
        df (pd.DataFrame): The dataset to be prepared.

    Returns:
        X (scipy.sparse.csr.csr_matrix): The feature matrix.
    """
    # Apply document preprocessing
    df['Job Description'] = df['Job Description'].apply(preprocess_document)
    df['Responsibilities'] = df['Responsibilities'].apply(preprocess_document)
    
    # TF-IDF vecotrization cannot be applied to tokenized data
    text_feature_columns = ['Job Description', 'Responsibilities']
    for column in text_feature_columns:
        df[column] = df[column].apply(lambda x: ' '.join(x))
    
    # Vectorize text features
    text_transformer = TfidfVectorizer()
    text_features = hstack([text_transformer.fit_transform(df[column]) for column in text_feature_columns])
    
    # One-hot encode categorical columns
    categorical_feature_columns = ['Qualifications', 'Work Type', 'Preference', 'Job Title', 'Role']
    categorical_transformer = OneHotEncoder()
    categorical_features = categorical_transformer.fit_transform(df[categorical_feature_columns])
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Company Size']])
    
    # Concatenate all feature vectors
    X = hstack([text_features, categorical_features, scaled_features])
    
    return X