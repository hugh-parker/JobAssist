import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

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

def prepare_dataset(df, text_transformer, categorical_transformer, scaler, text_feature_columns, categorical_feature_columns, numerical_feature_columns):
    """Prepare the dataset for training a machine learning model.

    Args:
        df (pd.DataFrame): The dataset to be prepared.
        text_transformer (TfidfVectorizer): The text vectorizer to be used.
        categorical_transformer (OneHotEncoder): The categorical encoder to be used.
        scaler (StandardScaler): The numerical feature scaler to be used.
        text_feature_columns (list): The text feature columns.
        categorical_feature_columns (list): The categorical feature columns.
        numerical_feature_columns (list): The numerical feature columns.

    Returns:
        X (scipy.sparse.csr.csr_matrix): The feature matrix.
    """
    # Vectorize text features
    text_features = hstack([text_transformer.transform(df[column]) for column in text_feature_columns])
    
    # One-hot encode categorical columns
    categorical_features = categorical_transformer.transform(df[categorical_feature_columns])
    
    # Scaling numerical features
    scaled_features = scaler.transform(df[numerical_feature_columns])

    # Concatenate all feature vectors
    X = hstack([text_features, categorical_features, scaled_features])
    
    return X

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