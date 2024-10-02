# 🧑‍💻 Resume Parser & Job Recommendation Chatbot

This project is an **AI-powered resume parser and job recommendation chatbot** that helps job seekers find suitable positions based on their skills and qualifications. The system parses resumes, extracts skills, and matches them with job descriptions using machine learning techniques. The project is built using **Python**, **Streamlit**, and **NLP models** for a seamless experience.

---

## 🚀 Features

- **Resume Parsing**: Extracts text and relevant skills from resumes in various formats (`.pdf`, `.docx`, `.txt`).
- **Skill Extraction**: Uses NLP techniques to identify key skills from resumes and compare them to a pre-defined skillset.
- **Job Matching**: Recommends the most suitable job openings from a database using **TF-IDF** and **Nearest Neighbors** algorithms.
- **Chatbot Interface**: A conversational interface to interact with the system, allowing users to query for top jobs, hiring companies, and job locations.
- **Streamlit Web App**: A simple and interactive web application for uploading resumes and viewing job matches.

---

## 🛠️ Tech Stack

- **Python**: Core language for all logic and processing.
- **Streamlit**: Web framework for building the user interface.
- **NLTK**: Natural Language Toolkit for skill extraction and text processing.
- **TF-IDF & Nearest Neighbors**: Algorithms for job matching and ranking.
- **PyPDF2, python-docx**: Libraries for reading resumes in PDF and Word formats.
- **pandas**: For data manipulation and handling job datasets.
- **scikit-learn**: Machine learning algorithms for text vectorization and job matching.

---

## 📂 Project Structure

```
├── data/
│   └── job_final.csv            # Job dataset containing job descriptions, companies, and locations.
├── scripts/
│   └── app.py                   # Main application file with Streamlit setup.
├── uploads/
│   └── uploaded_resume.*        # Placeholder for uploaded resumes.
├── README.md                    # Project documentation.
├── requirements.txt             # Dependencies for the project.
```

---

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/resume-parser-job-recommendation-chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd resume-parser-job-recommendation-chatbot
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK stopwords:

   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

---

## 🚀 Running the Application

1. Ensure all dependencies are installed.
2. Start the Streamlit app by running:

   ```bash
   streamlit run scripts/app.py
   ```

3. Upload a resume in `.pdf`, `.docx`, or `.txt` format.
4. View extracted skills and the top job matches!

---

## 🧠 How It Works

1. **Upload a Resume**: Upload your resume in one of the supported formats (`.pdf`, `.docx`, or `.txt`).
2. **Skill Extraction**: The system uses basic NLP techniques to extract skills from the resume text.
3. **Job Matching**: Your skills are matched with the job descriptions in the dataset using **TF-IDF vectorization** and **Nearest Neighbors**.
4. **Chatbot Interaction**: You can interact with the chatbot to query specific job-related information like top jobs, hiring companies, or job locations for a particular skill.

---

## 📊 Dataset

The job dataset (`job_final.csv`) includes the following columns:

- **Position**: Job title.
- **Company**: Company offering the job.
- **Location**: Job location.
- **Job_Description**: Detailed description of the role.

---

## 💡 Future Improvements

- **Expand Skill Database**: Add more domain-specific skills for better skill extraction.
- **Improve Matching Algorithm**: Integrate more advanced matching techniques, such as deep learning models, for better recommendations.
- **Expand Job Dataset**: Add more diverse job data to increase the scope of recommendations.
- **User Authentication**: Implement user login and profile features to save and track job search history.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the system or add new features, feel free to submit a pull request. Please make sure to follow the contribution guidelines.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- Thanks to the creators of **Streamlit**, **NLTK**, and **scikit-learn** for providing the tools to build this system.
- Special thanks to the open-source community for constant support and inspiration.

---