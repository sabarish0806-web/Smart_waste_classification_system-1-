import os
import io
import pytest
from PIL import Image
from app import create_app
from models import db, ClassificationLog

@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "UPLOAD_FOLDER": os.path.join(os.path.dirname(__file__), 'test_uploads')
    })
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def create_mock_image_file(filename="test.jpg", format="JPEG", color=(200, 50, 50)):
    img = Image.new("RGB", (200, 200), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return (img_bytes, filename)

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Smart Waste AI' in response.data

def test_get_guidance_endpoint(client):
    response = client.get('/api/guidance')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'Plastic' in data['categories']
    assert 'Paper' in data['categories']

def test_classify_valid_image(client):
    img_bytes, filename = create_mock_image_file()
    data = {'file': (img_bytes, filename)}
    
    response = client.post('/api/classify', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    res = response.get_json()
    assert res['success'] is True
    assert 'category' in res
    assert 'confidence' in res
    assert 'guidance' in res
    assert 'is_low_confidence' in res

def test_classify_invalid_extension(client):
    data = {'file': (io.BytesIO(b"fake data"), "document.pdf")}
    response = client.post('/api/classify', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    res = response.get_json()
    assert 'error' in res

def test_analytics_and_history_endpoints(client):
    # Perform two classifications
    img1, name1 = create_mock_image_file("item1.png", "PNG", (50, 200, 50))
    img2, name2 = create_mock_image_file("item2.jpg", "JPEG", (50, 50, 200))

    client.post('/api/classify', data={'file': (img1, name1)}, content_type='multipart/form-data')
    client.post('/api/classify', data={'file': (img2, name2)}, content_type='multipart/form-data')

    # Test history
    hist_res = client.get('/api/history')
    assert hist_res.status_code == 200
    hist_data = hist_res.get_json()
    assert hist_data['success'] is True
    assert len(hist_data['history']) == 2

    # Test analytics
    analytics_res = client.get('/api/analytics')
    assert analytics_res.status_code == 200
    analytics_data = analytics_res.get_json()
    assert analytics_data['success'] is True
    assert analytics_data['total_scans'] == 2
    assert analytics_data['avg_confidence'] > 0
