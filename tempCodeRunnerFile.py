def get_job_recommendations(resume_text, precomputed_embeddings):
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     resume_embedding = model.encode([preprocess_text(resume_text)])
#     # Assuming resume_embedding is the result from model.encode([preprocess_text(resume_text)])
#     if resume_embedding.ndim == 1:  # If somehow it's a 1D array
#         resume_embedding = np.reshape(resume_embedding, (1, -1))
#     elif resume_embedding.ndim > 2:  # If it's more than 2D for some reason
#         resume_embedding = np.reshape(resume_embedding, (resume_embedding.shape[0], -1))
#     cosine_similarities = cosine_similarity(resume_embedding, precomputed_embeddings)
#     top_five = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)[:5]

#     df = pd.read_csv(JOB_DESCRIPTIONS_CSV_PATH)
#     print("1: " + resume_embedding.shape)
#     print("2: " + job_descriptions_embeddings.shape)
#     return [{
#         'company': df.iloc[i]['company'],
#         'jobtitle': df.iloc[i]['jobtitle'],
#         'jobdescription': df.iloc[i]['jobdescription'][40:200],  # Snippet
#         'similarity_score': round(score * 100, 2),
#     } for i, score in top_five]