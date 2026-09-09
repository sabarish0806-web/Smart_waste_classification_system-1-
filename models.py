from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ClassificationLog(db.Model):
    __tablename__ = 'classification_logs'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(512), nullable=False)
    category = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_low_confidence = db.Column(db.Boolean, default=False)
    processing_time_ms = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'image_url': f'/uploads/{self.filename}',
            'category': self.category,
            'confidence': round(self.confidence, 1),
            'is_low_confidence': self.is_low_confidence,
            'processing_time_ms': round(self.processing_time_ms, 1) if self.processing_time_ms else None,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
