import streamlit as st
from docx import Document
import PyPDF2
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from ftfy import fix_text
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Load NLTK stopwords
nltk.download('stopwords')
stopw = set(stopwords.words('english'))

# Load the job dataset
df = pd.read_csv('job_final.csv')

# Preprocess job descriptions for TF-IDF matching
df['test'] = df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word not in stopw]))

# N-gram function to extract features
def ngrams(string, n=3):
    string = fix_text(string)  # Fix text
    string = string.encode("ascii", errors="ignore").decode()  # Remove non-ascii chars
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # Normalize case
    string = re.sub(' +', ' ', string).strip()  # Remove multiple spaces
    string = ' ' + string + ' '  # Pad names for n-grams
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# TF-IDF Vectorizer and Nearest Neighbors
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tfidf = vectorizer.fit_transform(df['test'].values.astype('U'))
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)

# Function to extract skills from resume (basic extraction from text)
def extract_skills_from_text(text):
    # Define a list of possible skills (this list can be customized)
    skills_db = ['python', 'java', 'data analysis', 'machine learning', 'deep learning', 'nlp', 'sql', 'excel', 'power bi']
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words from text
    extracted_skills = [skill for skill in skills_db if skill in words]
    return extracted_skills

# Function to read resume from .txt, .docx, or .pdf files
def extract_resume_details(file_path):
    text = ""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
    elif file_path.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    return text

# Function to find nearest jobs based on skills
def getNearestN(query):
    queryTFIDF_ = vectorizer.transform(query)
    distances, indices = nbrs.kneighbors(queryTFIDF_)
    return distances, indices

# Streamlit App
st.title("Job Recommendation Chatbot")

# Upload Resume
uploaded_file = st.file_uploader("Upload your resume (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    with open(os.path.join("uploaded_resume." + uploaded_file.name.split('.')[-1]), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract resume details
    resume_text = extract_resume_details(f"uploaded_resume.{uploaded_file.name.split('.')[-1]}")
    if resume_text:
        # Extract skills from resume
        st.write("Resume Skills Extracted:")
        resume_skills = extract_skills_from_text(resume_text)
        st.write(resume_skills)

        skills = []
        skills.append(' '.join(word for word in resume_skills))

        # Find matching jobs
        distances, indices = getNearestN(skills)
        matches = []
        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)
            temp = [dist]
            matches.append(temp)

        matches = pd.DataFrame(matches, columns=['Match confidence'])
        df['match'] = matches['Match confidence']
        df1 = df.sort_values('match')
        top_jobs = df1[['Position', 'Company', 'Location']].head(10).reset_index(drop=True)

        # Show top job matches
        st.write("Top Job Matches for Your Profile:")
        st.dataframe(top_jobs)

# User Input for Chatbot Interaction
st.header("Ask Me About Jobs!")
user_input = st.text_input("Enter your query (e.g., 'Top jobs for Python', 'Top hirers in data science', 'Locations for ML roles'):")

if user_input:
    # Process user queries
    if 'top jobs' in user_input.lower():
        skill_query = user_input.split('for')[-1].strip()
        st.write(f"Fetching top jobs for the skill: {skill_query}")
        filtered_jobs = df[df['test'].str.contains(skill_query, case=False, na=False)]
        if not filtered_jobs.empty:
            st.dataframe(filtered_jobs[['Position', 'Company', 'Location']].head(10).reset_index(drop=True))
        else:
            st.write("No jobs found for the specified skill.")
    elif 'top hirers' in user_input.lower():
        skill_query = user_input.split('in')[-1].strip()
        st.write(f"Fetching top hirers for the skill: {skill_query}")
        filtered_jobs = df[df['test'].str.contains(skill_query, case=False, na=False)]
        top_hirers = filtered_jobs['Company'].value_counts().head(10).reset_index()
        top_hirers.columns = ['Company', 'Job Count']
        st.dataframe(top_hirers)
    elif 'locations' in user_input.lower():
        role_query = user_input.split('for')[-1].strip()
        st.write(f"Fetching locations for the role: {role_query}")
        filtered_jobs = df[df['Position'].str.contains(role_query, case=False, na=False)]
        top_locations = filtered_jobs['Location'].value_counts().head(10).reset_index()
        top_locations.columns = ['Location', 'Job Count']
        st.dataframe(top_locations)
    else:
        st.write("Sorry, I didn't understand that query. Please try another question!")

st.write("Thank you for using the Job Recommendation Chatbot!")
