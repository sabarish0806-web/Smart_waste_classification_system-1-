import os
import tempfile

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Check if running in a cloud container (Render / Heroku)
IS_CLOUD = os.environ.get('RENDER') or os.environ.get('PORT')

if IS_CLOUD:
    TEMP_DIR = tempfile.gettempdir()
    UPLOAD_DIR = os.path.join(TEMP_DIR, 'uploads')
    DB_PATH = os.path.join(TEMP_DIR, 'waste_classification.db')
else:
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    DB_PATH = os.path.join(BASE_DIR, 'waste_classification.db')

os.makedirs(UPLOAD_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smart-waste-classification-secret-key-2026')
    UPLOAD_FOLDER = UPLOAD_DIR
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit as per FR-1
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Classification Thresholds
    LOW_CONFIDENCE_THRESHOLD = 60.0  # Percentage threshold for low-confidence warning (FR-6)
