import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("job_descriptions.csv")  # Update filename if necessary

df_jobs = load_data()

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"[^\w\s]", "", text.lower())  # Remove punctuation & lowercase
    words = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in words if word not in stop_words])  # Remove stopwords

# Apply preprocessing to job descriptions
df_jobs["Cleaned_Description"] = df_jobs["Job Description"].apply(preprocess_text)

# List of skills
skills_list = [
    "python", "java", "c++", "javascript", "sql", "r", "machine learning",
    "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "data analysis", "data visualization", "pandas", "numpy",
    "matplotlib", "big data", "aws", "azure", "docker", "kubernetes",
    "html", "css", "react", "angular", "node.js", "django", "flask"
]

# Function to extract skills from job descriptions
def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

df_jobs["Extracted_Skills"] = df_jobs["Cleaned_Description"].apply(extract_skills)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(df_jobs["Cleaned_Description"])

# Function to recommend jobs based on resume text
def recommend_jobs(resume_text, top_n=5):
    resume_text = preprocess_text(resume_text)  # Preprocess input resume
    resume_vector = vectorizer.transform([resume_text])  # Convert resume to vector
    similarity_scores = cosine_similarity(resume_vector, job_vectors)  # Compute similarity
    job_indices = similarity_scores.argsort()[0][-top_n:][::-1]  # Get top job indices
    return df_jobs.iloc[job_indices][["Job Title", "Company", "Extracted_Skills"]]

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
            st.markdown(f"**Job Title:** {row['Job Title']}")
            st.markdown(f"**Company:** {row['Company']}")
            st.markdown(f"**Required Skills:** {', '.join(row['Extracted_Skills'])}")
            st.write("---")
    else:
        st.warning("⚠️ Please enter a resume or skills to proceed.")
