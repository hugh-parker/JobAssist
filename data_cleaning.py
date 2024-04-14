import string
import re
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix

def preprocess_document(document, lemmatizer=WordNetLemmatizer()):
    """Apply NLP preprocessing to a given document.

    Args:
        document (str): The document to be preprocessed.
        lemmatizer (WordNetLemmatizer): The lemmatizer to be used.
        
    Returns:
        str: The preprocessed document.
    """     
    # Convert to lower case
    document = document.lower()

    # # Remove stopwords
    document_tokens = word_tokenize(document)
    document_tokens = [word for word in document_tokens if word not in stopwords.words('english')]
    
    # Remove punctuation from tokens
    extended_punctuation = string.punctuation + '“”‘’—'
    document_tokens = [''.join(char for char in s if char not in extended_punctuation) for s in document_tokens]
    
    # Perform lemmatization
    document_tokens = [lemmatizer.lemmatize(word) for word in document_tokens]

    return ' '.join(document_tokens)

def prepare_job_description(df, text_transformer):
    """Prepare the job description column of the dataset.

    Args:
        df (_type_): _description_
        text_transformer (_type_): _description_
    """
    # Apply document preprocessing
    df['Job Description'] = df['Job Description'].apply(preprocess_document)
    
    # TF-IDF vecotrization cannot be applied to tokenized data
    df['Job Description'] = df['Job Description'].apply(lambda x: ' '.join(x))
    
    # Vectorize text features
    return hstack([text_transformer.transform(df['Job Description'])])

def prepare_dataset(df, text_transformer, categorical_transformer, scaler):
    """Prepare the dataset for training a machine learning model.

    Args:
        df (pd.DataFrame): The dataset to be prepared.
        text_transformer (TfidfVectorizer): The text vectorizer to be used.
        categorical_transformer (OneHotEncoder): The categorical encoder to be used.
        scaler (StandardScaler): The numerical feature scaler to be used.

    Returns:
        X (scipy.sparse.csr.csr_matrix): The feature matrix.
    """
    # Apply document preprocessing
    df['Job Description'] = df['Job Description'].apply(preprocess_document)
    df['Responsibilities'] = df['Responsibilities'].apply(preprocess_document)
    
    # Vectorize text features
    text_features = hstack([text_transformer.transform(df[column]) for column in text_feature_columns])
    
    # One-hot encode categorical columns
    categorical_feature_columns = ['Qualifications', 'Work Type', 'Preference', 'Job Title', 'Role']
    categorical_features = categorical_transformer.transform(df[categorical_feature_columns])
    
    # Scaling numerical features
    # scaled_features = scaler.transform(df[['Company Size']])
    scaled_features = csr_matrix(df[['Company Size']])

    # Concatenate all feature vectors
    X = hstack([text_features, categorical_features, scaled_features])
    
    return X

import re

def preprocess_skills(skills_str):
    """Preprocess the skills column.

    Args:
        skills_str (str): The skills string to be preprocessed.

    Returns:
        list: A list of preprocessed skills.
    """
    # Remove any parenthetical information
    skills_str = re.sub(r'\([^()]*\)', '', skills_str)
    
    # Split skills based on whitespace
    skills_list = skills_str.split()
    
    # Merge consecutive elements if they form a multi-word skill
    merged_skills = []
    current_skill = ""
    for word in skills_list:
        if word[0].isupper() and current_skill:
            merged_skills.append(current_skill.strip())
            current_skill = word
        else:
            current_skill += " " + word
    if current_skill:
        merged_skills.append(current_skill.strip())
    
    return merged_skills