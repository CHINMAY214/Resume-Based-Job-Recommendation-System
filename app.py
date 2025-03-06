from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load the dataset
def load_data():
    return pd.read_csv("job_descriptions1.csv")  # Update filename if necessary

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    resume_text = data.get("resume_text", "").strip()
    
    if not resume_text:
        return jsonify({"error": "Please enter a resume or skills to proceed."}), 400
    
    recommended_jobs = recommend_jobs(resume_text)
    jobs_list = recommended_jobs.to_dict(orient="records")
    return jsonify({"recommended_jobs": jobs_list})

if __name__ == "__main__":
    app.run(debug=True)
