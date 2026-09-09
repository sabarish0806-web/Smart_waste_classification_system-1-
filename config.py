import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smart-waste-classification-secret-key-2026')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit as per FR-1
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f"sqlite:///{os.path.join(BASE_DIR, 'waste_classification.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Classification Thresholds
    LOW_CONFIDENCE_THRESHOLD = 60.0  # Percentage threshold for low-confidence warning (FR-6)
