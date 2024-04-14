from sklearn.metrics.pairwise import cosine_similarity
import data_cleaning as dc

def simple_get_skills(df, job_description, tfidf_vectorizer, skill_amt=5):
    """Get the top n skills for a job description.

    Args:
        df (DataFrame): The DataFrame containing the job descriptions.
        job_description (str): The job description to analyze.
        tfidf_vectorizer (TfidfVectorizer): The TfidfVectorizer used to transform the job description.
        model (Model): The model used to predict the skills.
        skill_amt (int): The number of skills to return.
    """
    # Preprocess the input job description
    cleaned_input = dc.preprocess_document(job_description)
    
    # Transform the input into TF-IDF vector
    input_vector = tfidf_vectorizer.transform([cleaned_input])
    
    # Calculate cosine similarity between input and all job descriptions
    similarity_scores = cosine_similarity(input_vector, X)
    
    # Get top N most similar job descriptions
    top_similar_indices = similarity_scores.argsort()[0][-skill_amt:][::-1]
    
    # Extract skills from top similar job descriptions
    recommended_skills = set()
    for idx in top_similar_indices:
        skills = df.loc[idx, 'skills']
        for skill in skills:
            recommended_skills.add(skill)
    
    return recommended_skills

def model_get_skills(job_description, vectorizer, mlb, knn_model):
    """
    Takes a raw job description, vectorizes it, predicts skills using a KNN model,
    and returns the predicted skills as a list.

    Args:
        job_description (str): The raw job description to predict skills for.
        vectorizer (TfidfVectorizer): The TfidfVectorizer used to transform the job description.
        mlb (MultiLabelBinarizer): The MultiLabelBinarizer used to transform the predicted skills.
        knn_model (KNeighborsClassifier): The KNN model used to predict the skills.
    
    Returns:
        list: The predicted skills for the job description.
    """
    # Step 1: Preprocess the job description using the same vectorizer used during training
    processed_description = vectorizer.transform([job_description])
    
    # Step 2: Use the trained KNN model to predict the skills
    predicted_skills_binary = knn_model.predict(processed_description)
    
    # Step 3: Inverse transform the binary predictions back to skills using the label binarizer
    predicted_skills = mlb.inverse_transform(predicted_skills_binary)
    
    # Step 4: Convert the tuple of skills to a list (since inverse_transform returns a list of tuples)
    return list(predicted_skills[0])