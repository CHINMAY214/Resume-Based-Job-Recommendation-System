import os
import streamlit as st
import kagglehub
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to download dataset from Kaggle
def download_dataset():
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        st.info("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")
        st.success("Dataset downloaded successfully!")
        return path
    return dataset_path

# Download dataset
dataset_path = download_dataset()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(dataset_path)

df = load_data()

# Load NLP model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Preprocess job descriptions
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

df["cleaned_description"] = df["job_description"].astype(str).apply(preprocess_text)

# Vectorize job descriptions
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(df["cleaned_description"])

# Function to recommend jobs
def recommend_jobs(user_skills, top_n=5):
    user_skills_cleaned = preprocess_text(user_skills)
    user_vector = vectorizer.transform([user_skills_cleaned])
    similarities = cosine_similarity(user_vector, job_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return df.iloc[top_indices][["job_title", "company", "location", "job_description"]]

# Streamlit UI
st.title("🔍 Resume-Based Job Recommendation System")
st.write("Enter your skills to find matching job postings.")

user_input = st.text_area("📝 Enter your skills (e.g., Python, Data Analysis, Machine Learning)", "")

if st.button("Recommend Jobs"):
    if user_input:
        recommendations = recommend_jobs(user_input)
        st.write("### Recommended Jobs for You:")
        st.dataframe(recommendations)
    else:
        st.warning("Please enter your skills to get recommendations.")

st.write("📊 Dataset Overview")
st.write(df.head())
