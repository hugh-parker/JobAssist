import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

# Load the dataset
df = pd.read_csv("/Users/hughparker/Desktop/JobAssist/job_descriptions.csv", nrows=40000)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)

# Preprocessing for skills column
def preprocess_skills(skills):
    skills = skills.lower()
    skills = skills.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return skills

# Apply preprocessing to skills column
df['cleaned_skills'] = df['skills'].apply(preprocess_skills)

df['cleaned_job_description'] = df['Job Description'].apply(preprocess_text)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_job_description'])

# Skill Recommendation Model
def recommend_skills(job_description, top_n=5):
    # Preprocess the input job description
    cleaned_input = preprocess_text(job_description)
    
    # Transform the input into TF-IDF vector
    input_vector = tfidf_vectorizer.transform([cleaned_input])
    
    # Calculate cosine similarity between input and all job descriptions
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
    
    # Get top N most similar job descriptions
    top_similar_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    
    # Extract skills from top similar job descriptions
    recommended_skills = set()
    for idx in top_similar_indices:
        # Split skills by capital letters instead of commas
        skills = df.loc[idx, 'skills']
        current_skill = []
        for char in skills:
            if char.isupper():
                recommended_skills.add("".join(current_skill).strip())
                current_skill = [char]
            else:
                current_skill.append(char)
        if current_skill:
            recommended_skills.add("".join(current_skill).strip())
    
    return recommended_skills

# Test the recommendation model
job_description = "Seeking a teacher / educator with experience working with students and organizing schedules and curriculum."
recommended_skills = recommend_skills(job_description)
print("Recommended Skills:", recommended_skills)
