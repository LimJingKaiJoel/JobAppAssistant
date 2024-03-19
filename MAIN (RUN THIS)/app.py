from flask import Flask, request, render_template
import os
import fitz  # PyMuPDF for extracting text from PDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='../templates')

# CHANGE THIS TO YOUR RESUME
UPLOAD_FOLDER = 'test-resume'
# CHANGE THIS TO YOUR JOB DESCRIPTION TEST SET (WEB-SCRAPED DATA)
JOB_DESCRIPTIONS_CSV_PATH = '../web_scraped_data/data.csv'
# Directory to store precomputed embeddings
PRECOMPUTED_DATA_DIR = 'precomputed_data'
# column name for your dataset
JOB_TITLE = 'Title'
# File path for precomputed embeddings
EMBEDDINGS_FILE_miniLM = os.path.join(PRECOMPUTED_DATA_DIR, 'job_descriptions_embeddings_miniLM.joblib')
EMBEDDINGS_FILE_distilbert = os.path.join(PRECOMPUTED_DATA_DIR, 'job_descriptions_embeddings_distilbert.joblib')

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRECOMPUTED_DATA_DIR, exist_ok=True)

# this function does nlp standard text preprocessing, word_level tokenization, lemmatization and removing of stopwords
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set)

# this function applies the preprocessing function to job descriptions, and saves the embeddings of embedding models to a file
def preprocess_and_save_embeddings(csv_path, embeddings_path1, embeddings_path2):
    # only runs if embedding_path doesn't exist
    if not os.path.exists(embeddings_path1) or not os.path.exists(embeddings_path2):
        df = pd.read_csv(csv_path)
        df[JOB_TITLE] = df[JOB_TITLE].fillna('')
        
        # change this line to remove text preprocessing for embedding models
        descriptions = df[JOB_TITLE].apply(preprocess_text).tolist()
        
        model1 = SentenceTransformer('all-MiniLM-L6-v2')
        model2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        embeddings1 = model1.encode(descriptions)
        embeddings2 = model2.encode(descriptions)
        
        joblib.dump(embeddings1, embeddings_path1)
        joblib.dump(embeddings2, embeddings_path2)

        print(f"Embeddings saved to {embeddings_path1}\nEmbeddings also saved to {embeddings_path2}")

# this function checks if the filename is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# this function uses fitz module to extract text from pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# function to save vectorizer and tfidf_matrix to precomputed data folder
def preprocess_and_save_job_descriptions(csv_path, save_path):
    df = pd.read_csv(csv_path)
    df[JOB_TITLE] = df[JOB_TITLE].fillna('')

    # change this to remove preprocessing for tf_idf, but i think should keep just for tf-idf
    job_descriptions = df[JOB_TITLE].apply(preprocess_text).tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    
    # Save the vectorizer and tfidf_matrix for later use
    joblib.dump(vectorizer, os.path.join(save_path, 'vectorizer.joblib'))
    joblib.dump(tfidf_matrix, os.path.join(save_path, 'tfidf_matrix.joblib'))

# call function to save vectorizer and tfidf_matrix to precomputed data folder
preprocess_and_save_job_descriptions(JOB_DESCRIPTIONS_CSV_PATH, PRECOMPUTED_DATA_DIR)

# function to return vectorizer and tfidf_matrix that was precomputed
def load_precomputed_data(save_path):
    vectorizer = joblib.load(os.path.join(save_path, 'vectorizer.joblib'))
    tfidf_matrix = joblib.load(os.path.join(save_path, 'tfidf_matrix.joblib'))
    return vectorizer, tfidf_matrix

# call function to get vectorizer and precomputed tfidf matrix
vectorizer, precomputed_tfidf_matrix = load_precomputed_data(PRECOMPUTED_DATA_DIR)

# call function to save embeddings for embedding models, and load it into variable job_descriptions_embeddings
preprocess_and_save_embeddings(JOB_DESCRIPTIONS_CSV_PATH, EMBEDDINGS_FILE_miniLM, EMBEDDINGS_FILE_distilbert)
job_descriptions_embeddings1 = joblib.load(EMBEDDINGS_FILE_miniLM)
job_descriptions_embeddings2 = joblib.load(EMBEDDINGS_FILE_distilbert)

# can don't call this for the bagged model
def get_top_contributing_words_direct(vectorizer, target_vector, comparison_vector, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    # ensure both vectors are in a flattened array form for direct comparison
    target_vector_flat = target_vector.toarray().flatten()
    comparison_vector_flat = comparison_vector.toarray().flatten()
    # calculate contribution scores by element-wise multiplication of the vectors
    contribution_scores = target_vector_flat * comparison_vector_flat
    # Identify the indices of the top 'n' contributing features
    top_indices = contribution_scores.argsort()[-top_n:][::-1]
    # Extract the top contributing words and their scores
    top_words = [(feature_names[i], contribution_scores[i]) for i in top_indices if contribution_scores[i] > 0]
    return top_words

def get_job_recommendations(resume_text, precomputed_embedding1, precomputed_embedding2, df, vectorizer, tfidf_matrix):
    # getting embeddings for both embedding models 
    model1 = SentenceTransformer('all-MiniLM-L6-v2')
    model2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    resume_embedding1 = model1.encode([preprocess_text(resume_text)])
    resume_embedding2 = model2.encode([preprocess_text(resume_text)])
    
    # getting resume_tfidf for tfidf model
    preprocessed_resume = preprocess_text(resume_text)
    resume_tfidf = vectorizer.transform([preprocessed_resume])

    # Assuming resume_embedding is the result from model.encode([preprocess_text(resume_text)])
    # if resume_embedding.ndim == 1:  # If somehow it's a 1D array
    #     resume_embedding = np.reshape(resume_embedding, (1, -1))
    # elif resume_embedding.ndim > 2:  # If it's more than 2D for some reason
    #     resume_embedding = np.reshape(resume_embedding, (resume_embedding.shape[0], -1))

    # 1 == miniLM, 2 == distilbert, 3 == tdidf
    cosine_similarities1 = cosine_similarity(resume_embedding1, precomputed_embedding1)
    cosine_similarities2 = cosine_similarity(resume_embedding2, precomputed_embedding2)
    cosine_similarities3 = cosine_similarity(resume_tfidf, tfidf_matrix)

    # ensure that these 3 are the same shapes for the flatten and normalisation to work
    # print(cosine_similarities1.shape)
    # print(cosine_similarities2.shape)
    # print(cosine_similarities3.shape)

    # normalise the cosine similarity scores for all models
    scaler = MinMaxScaler()

    # NOTE THAT THIS IS 33% WEIGHTAGE TO TFIDF'S COSINE SIMILARITY. CAN CHANGE TO 50-50 WITH EMBEDDING MODEL
    all_cosine_similarities = np.vstack((cosine_similarities1.flatten(), cosine_similarities2.flatten(), cosine_similarities3.flatten()))
    normalized_cosine_similarities = scaler.fit_transform(all_cosine_similarities.T).T

    # get averaged of normalised cosine similarities
    bagged_cosine_similarities = np.mean(normalized_cosine_similarities, axis=0)

    # reshape back to the original shape if necessary
    bagged_cosine_similarities = bagged_cosine_similarities.reshape(cosine_similarities1.shape)

    top_five = sorted(enumerate(bagged_cosine_similarities[0]), key=lambda x: x[1], reverse=True)[:5]

    return [{
        'company': df.iloc[i]['Company'],
        'jobtitle': df.iloc[i][JOB_TITLE],
        'jobdescription': df.iloc[i]['Job Link'],  # snippet, to change based on how you want to display, can display link too
        'similarity_score': round(score * 100, 2),
    } for i, score in top_five]

# Load teacher and student models
teacher_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
student_model = SentenceTransformer('all-MiniLM-L6-v2')  # Assuming this is your student model
import time

def model_evaluation_with_user_input(input_text):
    start_time_teacher = time.time()
    teacher_embedding = teacher_model.encode([input_text], convert_to_tensor=True)
    teacher_time = time.time() - start_time_teacher

    start_time_student = time.time()
    student_embedding = student_model.encode([input_text], convert_to_tensor=True)
    student_time = time.time() - start_time_student

    # For simplicity, let's assume we compare based on the computation time here
    # In a real scenario, you'd also look at the quality of embeddings or task-specific performance metrics
    evaluation_result = {
        'teacher_time': teacher_time,
        'student_time': student_time,
        'efficiency_improvement': teacher_time - student_time
    }
    return evaluation_result

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return 'No file part or invalid file extension', 400
        
        # #just added this experience level
        # if request.form.get('experienceLevel'):
        #     experience_level = request.form.get('experienceLevel')
        
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        resume_text = extract_text_from_pdf(filename)
        # Perform model evaluation with the extracted text
        evaluation_result = model_evaluation_with_user_input(resume_text)
        # tfidf handling
        df = pd.read_csv(JOB_DESCRIPTIONS_CSV_PATH)
        #print(df)

        # call main function
        recommendations = get_job_recommendations(resume_text, job_descriptions_embeddings1, job_descriptions_embeddings2, df, vectorizer, precomputed_tfidf_matrix)
        return render_template('display2.html', recommendations=recommendations, evaluation_result=evaluation_result)
    
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
