import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("job_descriptions1.csv")  # Update the filename if necessary

df_jobs = load_data()

# Load NLP model
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation & lowercase
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])  # Lemmatization

# Apply preprocessing to job descriptions
df_jobs['Cleaned_Description'] = df_jobs['Job Description'].apply(preprocess_text)

# Expanded list of skills
skills_list = [
    "python", "java", "c++", "javascript", "sql", "r", "machine learning",
    "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
    "data analysis", "data visualization", "pandas", "numpy", "matplotlib", "big data",
    "apache spark", "hadoop", "hive", "aws", "azure", "google cloud", "docker",
    "kubernetes", "terraform", "ci/cd", "jenkins", "github actions", "html",
    "css", "react", "angular", "vue.js", "node.js", "express.js", "django",
    "flask", "spring boot", "mysql", "postgresql", "mongodb", "firebase",
    "rest api", "graphql", "microservices", "excel", "power bi", "tableau",
    "looker", "business intelligence", "forecasting", "financial modeling",
    "market research", "growth analytics", "ethical hacking", "penetration testing",
    "network security", "cyber threat intelligence", "siem", "soc"
]

# Function to extract skills from job descriptions
def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

df_jobs['Extracted_Skills'] = df_jobs['Cleaned_Description'].apply(extract_skills)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(df_jobs['Cleaned_Description'])

# Function to recommend jobs based on resume text
def recommend_jobs(resume_text, top_n=5):
    resume_text = preprocess_text(resume_text)  # Preprocess input resume
    resume_vector = vectorizer.transform([resume_text])  # Convert resume to vector

    similarity_scores = cosine_similarity(resume_vector, job_vectors)  # Compute similarity
    job_indices = similarity_scores.argsort()[0][-top_n:][::-1]  # Get top job indices

    return df_jobs.iloc[job_indices][['Job Title', 'Company', 'Extracted_Skills']]

# Streamlit App Interface
st.title("🔍 Resume-Based Job Recommendation System")
st.write("Upload your resume or enter your skills below to find the best-matching jobs.")

# Resume Input
resume_text = st.text_area("📜 Paste Your Resume or Enter Skills Here:", "")

if st.button("Find Jobs"):
    if resume_text.strip():
        recommended_jobs = recommend_jobs(resume_text)
        st.subheader("💼 Recommended Jobs")
        for _, row in recommended_jobs.iterrows():
            st.markdown(f"**Job Title:** {row['Job_Title']}")
            st.markdown(f"**Company:** {row['Company']}")
            st.markdown(f"**Required Skills:** {', '.join(row['Extracted_Skills'])}")
            st.write("---")
    else:
        st.warning("⚠️ Please enter a resume or skills to proceed.")
