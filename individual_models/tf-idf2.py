from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF for extracting text from PDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import joblib  # For serialization

app = Flask(__name__)

# Define the folder where uploaded resumes will be stored
UPLOAD_FOLDER = 'test-resume'
PRECOMPUTED_DATA_FOLDER = 'precomputed_data'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Specify the path to your CSV file containing job descriptions
JOB_DESCRIPTIONS_CSV_PATH = 'download_this_dataset/jd.csv'

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    text = str(text) if text is not None else ''
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    return ' '.join(cleaned_tokens)


def preprocess_and_save_job_descriptions(csv_path, save_path):
    df = pd.read_csv(csv_path)
    df['jobdescription'] = df['jobdescription'].fillna('')
    job_descriptions = df['jobdescription'].apply(preprocess_text).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    
    # Save the vectorizer and tfidf_matrix for later use
    joblib.dump(vectorizer, os.path.join(save_path, 'vectorizer.joblib'))
    joblib.dump(tfidf_matrix, os.path.join(save_path, 'tfidf_matrix.joblib'))
    
preprocess_and_save_job_descriptions(JOB_DESCRIPTIONS_CSV_PATH, PRECOMPUTED_DATA_FOLDER)

def load_precomputed_data(save_path):
    vectorizer = joblib.load(os.path.join(save_path, 'vectorizer.joblib'))
    tfidf_matrix = joblib.load(os.path.join(save_path, 'tfidf_matrix.joblib'))
    return vectorizer, tfidf_matrix

# Load precomputed data at app start
vectorizer, precomputed_tfidf_matrix = load_precomputed_data(PRECOMPUTED_DATA_FOLDER)

def load_job_descriptions(csv_path):
    return pd.read_csv(csv_path)

def get_top_contributing_words_direct(vectorizer, target_vector, comparison_vector, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    # Ensure both vectors are in a flattened array form for direct comparison
    target_vector_flat = target_vector.toarray().flatten()
    comparison_vector_flat = comparison_vector.toarray().flatten()
    # Calculate contribution scores by element-wise multiplication of the vectors
    contribution_scores = target_vector_flat * comparison_vector_flat
    # Identify the indices of the top 'n' contributing features
    top_indices = contribution_scores.argsort()[-top_n:][::-1]
    # Extract the top contributing words and their scores
    top_words = [(feature_names[i], contribution_scores[i]) for i in top_indices if contribution_scores[i] > 0]
    return top_words

def get_job_recommendations(resume_text, df, vectorizer, tfidf_matrix):
    preprocessed_resume = preprocess_text(resume_text)
    resume_tfidf = vectorizer.transform([preprocessed_resume])
    cosine_similarities = cosine_similarity(resume_tfidf, tfidf_matrix)
    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)[:5]
    
    top_recommendations = []
    for index, score in recommendations:
        # Direct comparison of the resume TF-IDF vector with a job description TF-IDF vector
        job_desc_tfidf_vector = tfidf_matrix[index]
        top_words = get_top_contributing_words_direct(vectorizer, resume_tfidf, job_desc_tfidf_vector, top_n=10)
        top_recommendations.append({
            'company': df.iloc[index]['company'],
            'jobtitle': df.iloc[index]['jobtitle'],
            'jobdescription': df.iloc[index]['jobdescription'][40:200],
            'similarity_score': round(score * 100, 2),
            'top_words': top_words
        })
    
    return top_recommendations



def handle_resume_upload(file):
    if file.filename == '':
        return 'No selected file'
    if not allowed_file(file.filename):
        return 'Invalid file extension'
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    resume_text = extract_text_from_pdf(filename)
        
    df = load_job_descriptions(JOB_DESCRIPTIONS_CSV_PATH)
    recommendations = get_job_recommendations(resume_text, df, vectorizer, precomputed_tfidf_matrix)
    
    return recommendations

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return 'No file part'
        
        recommendations = handle_resume_upload(file)
        if isinstance(recommendations, str):
            return recommendations
        return render_template('display.html', recommendations=recommendations)
    
    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
