# JobAppAssistant
Job Application Assistant: bagged transformer-based huggingface embedding models with tf-idf to recommend web-scraped real-time jobs based on uploaded resume.

main file that's an ensemble of the rest is: job_app_asst.py
- be careful not to edit and push changes to main as it's working now

activate venv: 
source venv/bin/activate (for mac)
venv\Scripts\activate (for windows)

to use inside venv:
python --version
if the python version in your venv is not the same as the one that you are using: unalias python
all modules are downloaded inside venv / lib / python 3.11

miniLM.py : miniLM embedding model
distilbert.py : distilbert embedding model
tf-idf.py : tf-idf model (works good)

bagged_HF : bagged cosine similarity scores given by both miniLM.py and distilbert.py, normalised
job_app_asst : final model with all 3 models (bagged both embedding models with tf-idf score)

accuracy check on:
1. job_app_asst
2. tf-idf (to see if it's better without the embedding models)
3. 

the reason why cos similiarity scores are so high on the bagged embedding models is because they are normalised to 1 individually
ie. the highest score is set to 1 for each model then averaged

