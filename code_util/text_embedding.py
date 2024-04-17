import numpy as np
import data_cleaning as dc

from scipy.sparse import hstack, csr_matrix

def document_vector(doc, word_vectors):
    """Create document vectors by averaging word vectors.
    
    Args:
        doc (str): The document to be embedded.
        word_vectors (gensim.models.keyedvectors.KeyedVectors): The word vectors to be used.
    
    Returns:
        np.ndarray: The document vector."""
    # Remove out-of-vocabulary words
    words = doc.split()
    word_vectors = [word_vectors[word] for word in words if word in word_vectors]
    
    # Return a vector of zeros if no words are in the model
    if len(word_vectors) == 0:
        return np.zeros(word_vectors.vector_size)
    else: # Else return the mean of the word vectors
        return np.mean(word_vectors, axis=0)

def embed_w2v_dataframe(df, word_vectors, categorical_transformer, numerical_transformer, categorical_features, numerical_features):
    """Embed the text features in the dataset.

    Args:
        df (pd.DataFrame): The dataset to be embedded.
        word_vectors (_type_): The word vectors to be used.
        categorical_transformer (OneHotEncoder): The categorical encoder to be used.
        numerical_transformer (StandardScaler): The numerical feature scaler to be used.
        categorical_features (list): The categorical feature columns.
        numerical_features (list): The numerical feature columns.
        
    Returns:
        scipy.sparse.csr.csr_matrix: The feature matrix.
    """
    temp_df = df.copy()

    # Vectorize 'Job Description' and 'responsibilities' columns
    temp_df['job_description_vec'] = temp_df['Preprocessed Job Description'].apply(lambda x: document_vector(x, word_vectors))
    temp_df['responsibilities_vec'] = temp_df['Preprocessed Responsibilities'].apply(lambda x: document_vector(x, word_vectors))

    # Concatenate text features into a single matrix
    text_features = np.hstack((np.vstack(temp_df['job_description_vec']), np.vstack(temp_df['responsibilities_vec'])))
    
    # Ensure text features are in sparse format for efficient concatenation
    text_features_sparse = csr_matrix(text_features)

    # Transform the categorical and numerical features
    categorical_features_transformed = categorical_transformer.transform(temp_df[categorical_features])
    numerical_features_transformed = numerical_transformer.transform(temp_df[numerical_features])
    
    # Ensure numerical features are in sparse format
    numerical_features_sparse = csr_matrix(numerical_features_transformed)

    # Concatenate all features
    return hstack([text_features_sparse, categorical_features_transformed, numerical_features_sparse])

def embed_description(df, word_vectors):
    """Embed the job description in the dataset.

    Args:
        df (pd.DataFrame): The dataset to be embedded.
        word_vectors (_type_): The word vectors to be used.
        
    Returns:
        scipy.sparse.csr.csr_matrix: The feature matrix.
    """
    # Preprocess 'Job Description' column
    df['Job Description'] = df['Job Description'].apply(dc.preprocess_document)

    # Vectorize 'Job Description' column
    df['job_description_vec'] = df['Job Description'].apply(lambda x: document_vector(x, word_vectors))

    # Return the embedded job description
    return np.array(df['job_description_vec'].tolist())    