import os
import uuid
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from models import db, ClassificationLog
from utils import allowed_file, get_category_guidance, WASTE_CATEGORIES
from ml_engine import classifier

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)

    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    with app.app_context():
        try:
            db.create_all()
        except Exception as err:
            print(f"Database init warning: {err}")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.svg', mimetype='image/svg+xml')

    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/api/classify', methods=['POST'])
    def classify_image():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({
                'error': 'Invalid file format. Only JPG, JPEG, and PNG images are allowed.'
            }), 400

        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': 'File size exceeds maximum allowed limit of 10MB.'
            }), 400

        try:
            original_filename = secure_filename(file.filename)
            file_ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
            unique_filename = f"{uuid.uuid4().hex}_{int(datetime.now(timezone.utc).timestamp())}.{file_ext}"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(saved_path)

            # Perform classification via ML engine
            pred_result = classifier.predict(saved_path, original_filename=original_filename)
            category = str(pred_result['category'])
            confidence = float(pred_result['confidence'])
            processing_time_ms = float(pred_result['processing_time_ms'])

            # Check low confidence threshold (FR-6: <60%)
            is_low_confidence = bool(confidence < app.config['LOW_CONFIDENCE_THRESHOLD'])

            # Log classification result in database (FR-7)
            log_id = 1
            try:
                log_entry = ClassificationLog(
                    filename=unique_filename,
                    original_filename=original_filename,
                    image_path=saved_path,
                    category=category,
                    confidence=confidence,
                    is_low_confidence=is_low_confidence,
                    processing_time_ms=processing_time_ms
                )
                db.session.add(log_entry)
                db.session.commit()
                log_id = log_entry.id
            except Exception as db_err:
                print(f"DB Log Warning: {db_err}")

            guidance = get_category_guidance(category)

            return jsonify({
                'success': True,
                'log_id': log_id,
                'filename': unique_filename,
                'original_filename': original_filename,
                'image_url': f'/uploads/{unique_filename}',
                'category': category,
                'confidence': confidence,
                'is_low_confidence': is_low_confidence,
                'processing_time_ms': processing_time_ms,
                'guidance': guidance,
                'all_scores': pred_result['all_scores'],
                'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            })

        except Exception as e:
            return jsonify({'error': f'Classification error: {str(e)}'}), 500

    @app.route('/api/history', methods=['GET'])
    def get_history():
        try:
            logs = ClassificationLog.query.order_by(ClassificationLog.created_at.desc()).limit(20).all()
            return jsonify({
                'success': True,
                'history': [log.to_dict() for log in logs]
            })
        except Exception:
            return jsonify({'success': True, 'history': []})

    @app.route('/api/analytics', methods=['GET'])
    def get_analytics():
        try:
            logs = ClassificationLog.query.all()
            total_scans = len(logs)

            if total_scans == 0:
                return jsonify({
                    'success': True,
                    'total_scans': 0,
                    'avg_confidence': 0.0,
                    'low_confidence_scans': 0,
                    'low_confidence_rate': 0.0,
                    'category_distribution': {cat: 0 for cat in WASTE_CATEGORIES.keys()}
                })

            category_counts = {cat: 0 for cat in WASTE_CATEGORIES.keys()}
            total_conf = 0.0
            low_conf_count = 0

            for log in logs:
                total_conf += log.confidence
                if log.is_low_confidence:
                    low_conf_count += 1
                if log.category in category_counts:
                    category_counts[log.category] += 1
                else:
                    category_counts['Other/Unknown'] += 1

            avg_conf = total_conf / total_scans
            low_conf_rate = (low_conf_count / total_scans) * 100.0

            return jsonify({
                'success': True,
                'total_scans': total_scans,
                'avg_confidence': round(avg_conf, 1),
                'low_confidence_scans': low_conf_count,
                'low_confidence_rate': round(low_conf_rate, 1),
                'category_distribution': category_counts
            })
        except Exception:
            return jsonify({
                'success': True,
                'total_scans': 0,
                'avg_confidence': 0.0,
                'low_confidence_scans': 0,
                'low_confidence_rate': 0.0,
                'category_distribution': {cat: 0 for cat in WASTE_CATEGORIES.keys()}
            })

    @app.route('/api/guidance', methods=['GET'])
    def get_guidance():
        return jsonify({
            'success': True,
            'categories': WASTE_CATEGORIES
        })

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)