from setuptools import setup, find_packages
import subprocess
import sys

def install_spacy_model():
    """Download and install the spaCy model"""
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

# Call the function to install the spaCy model
install_spacy_model()

setup(
    name='chatbot_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pyresparser',
        'streamlit',
        'python-docx',
        'ftfy',
        'scikit-learn',
        'pandas',
        'numpy',
        'nltk',
        'pdfminer.six',
        'spacy',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'start-chatbot=app.chatbot:main',  # Change this if you use a different entry point
        ],
    },
)
