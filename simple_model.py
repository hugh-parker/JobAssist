import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

# Load the dataset
df = pd.read_csv("/Users/hughparker/Desktop/JobAssist/job_descriptions.csv", nrows=50000)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)

def preprocess_skills(skills_str):
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
        skills = preprocess_skills(df.loc[idx, 'skills'])
        for skill in skills:
            recommended_skills.add(skill)
    
    return recommended_skills

# Test the recommendation model (please for now excuse the code duplication loll)
job_description1 = "Seeking an experienced financial accountant to join our auditing department."
job_description2 = "Urgently hiring a educational specialist in early-childhood development and special education curriculums."
job_description3 = "Looking for a skilled backend software developer with experience in networks and database design, and object-oriented programming in Java, Python, etc."

recommended_skills1 = recommend_skills(job_description1)
recommended_skills2 = recommend_skills(job_description2)
recommended_skills3 = recommend_skills(job_description3)

print("Recommended Skills (Accountant):", recommended_skills1)
print("Recommended Skills (Teacher):", recommended_skills2)
print("Recommended Skills (Software Developer):", recommended_skills3)


