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

app = Flask(__name__)

# Define the folder where uploaded resumes will be stored
UPLOAD_FOLDER = '/Users/joellim/Desktop/job_test v3/test-resume'
# Path to the CSV file containing job descriptions
JOB_DESCRIPTIONS_CSV_PATH = '/Users/joellim/Desktop/job_test v3/download_this_dataset/jd.csv'
# Directory to store precomputed embeddings
EMBEDDINGS_DIR = '/Users/joellim/Desktop/job_test v3/precomputed_data'
# File path for precomputed embeddings
EMBEDDINGS_FILE_miniLM = os.path.join(EMBEDDINGS_DIR, 'job_descriptions_embeddings_miniLM.joblib')
EMBEDDINGS_FILE_distilbert = os.path.join(EMBEDDINGS_DIR, 'job_descriptions_embeddings_distilbert.joblib')

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    return ' '.join(lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set)

def preprocess_and_save_embeddings(csv_path, embeddings_path1, embeddings_path2):
    # only runs if embedding_path doesn't exist
    if not os.path.exists(embeddings_path1) or not os.path.exists(embeddings_path2):
        df = pd.read_csv(csv_path)
        df['jobdescription'] = df['jobdescription'].fillna('')
        descriptions = df['jobdescription'].apply(preprocess_text).tolist()
        
        model1 = SentenceTransformer('all-MiniLM-L6-v2')
        model2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        embeddings1 = model1.encode(descriptions)
        embeddings2 = model2.encode(descriptions)
        
        joblib.dump(embeddings1, embeddings_path1)
        joblib.dump(embeddings2, embeddings_path2)

        print(f"Embeddings saved to {embeddings_path1}\nEmbeddings also saved to {embeddings_path2}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)

# Load or precompute embeddings
preprocess_and_save_embeddings(JOB_DESCRIPTIONS_CSV_PATH, EMBEDDINGS_FILE_miniLM, EMBEDDINGS_FILE_distilbert)
job_descriptions_embeddings1 = joblib.load(EMBEDDINGS_FILE_miniLM)
job_descriptions_embeddings2 = joblib.load(EMBEDDINGS_FILE_distilbert)

# Assuming job_descriptions_embeddings is loaded from joblib.load(EMBEDDINGS_FILE)
# if job_descriptions_embeddings.ndim > 2:
#     job_descriptions_embeddings = np.reshape(job_descriptions_embeddings, (job_descriptions_embeddings.shape[0], -1))

def get_job_recommendations(resume_text, precomputed_embedding1, precomputed_embedding2):
    model1 = SentenceTransformer('all-MiniLM-L6-v2')
    model2 = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    resume_embedding1 = model1.encode([preprocess_text(resume_text)])
    resume_embedding2 = model2.encode([preprocess_text(resume_text)])

    # Assuming resume_embedding is the result from model.encode([preprocess_text(resume_text)])
    # if resume_embedding.ndim == 1:  # If somehow it's a 1D array
    #     resume_embedding = np.reshape(resume_embedding, (1, -1))
    # elif resume_embedding.ndim > 2:  # If it's more than 2D for some reason
    #     resume_embedding = np.reshape(resume_embedding, (resume_embedding.shape[0], -1))
    cosine_similarities1 = cosine_similarity(resume_embedding1, precomputed_embedding1)
    cosine_similarities2 = cosine_similarity(resume_embedding2, precomputed_embedding2)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Stack the two similarity scores vertically for the scaler to understand them
    all_cosine_similarities = np.vstack((cosine_similarities1.flatten(), cosine_similarities2.flatten()))
    # Fit the scaler on the combined set and then transform
    normalized_cosine_similarities = scaler.fit_transform(all_cosine_similarities.T).T
    # Now, the cosine similarity scores are normalized and we can take their average
    bagged_cosine_similarities = np.mean(normalized_cosine_similarities, axis=0)
    # Reshape back to the original shape if necessary
    bagged_cosine_similarities = bagged_cosine_similarities.reshape(cosine_similarities1.shape)

    top_five = sorted(enumerate(bagged_cosine_similarities[0]), key=lambda x: x[1], reverse=True)[:5]

    df = pd.read_csv(JOB_DESCRIPTIONS_CSV_PATH)
    return [{
        'company': df.iloc[i]['company'],
        'jobtitle': df.iloc[i]['jobtitle'],
        'jobdescription': df.iloc[i]['jobdescription'][40:200],  # Snippet
        'similarity_score': round(score * 100, 2),
    } for i, score in top_five]

@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return 'No file part or invalid file extension', 400
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        resume_text = extract_text_from_pdf(filename)
        recommendations = get_job_recommendations(resume_text, job_descriptions_embeddings1, job_descriptions_embeddings2)
        return render_template('display2.html', recommendations=recommendations)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
