import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'jobready-secret-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///jobready.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
