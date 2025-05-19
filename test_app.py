import os
import unittest
import tempfile
from io import BytesIO
import app as flask_app
from PIL import Image
import numpy as np

class TumorDetectProTestCase(unittest.TestCase):
    """Test cases for TumorDetect Pro application"""

    def setUp(self):
        """Set up test client and create test image"""
        self.app = flask_app.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Create a test image
        self.test_img_path = os.path.join(tempfile.gettempdir(), 'test_image.jpg')
        img = Image.new('RGB', (150, 150), color='white')
        img.save(self.test_img_path)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
    
    def test_index_page(self):
        """Test that index page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'TumorDetect Pro', response.data)
    
    def test_upload_no_file(self):
        """Test prediction endpoint with no file"""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'No file part')
    
    def test_upload_empty_filename(self):
        """Test prediction endpoint with empty filename"""
        response = self.client.post('/predict', data={
            'file': (BytesIO(), '')
        })
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'No selected file')
    
    def test_upload_invalid_extension(self):
        """Test prediction endpoint with invalid file extension"""
        response = self.client.post('/predict', data={
            'file': (BytesIO(b'test'), 'test.txt')
        })
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('error', json_data)
        self.assertEqual(json_data['error'], 'File type not allowed')
    
    def test_upload_valid_image(self):
        """Test prediction endpoint with valid image"""
        # This test will only work if the model exists
        # If the model doesn't exist, it will return an error
        with open(self.test_img_path, 'rb') as img:
            response = self.client.post('/predict', data={
                'file': (img, 'test_image.jpg')
            })
        
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        
        # If model exists, check for prediction results
        if 'error' not in json_data:
            self.assertIn('class', json_data)
            self.assertIn('confidence', json_data)
            self.assertIn('probabilities', json_data)
            self.assertIn('visualization', json_data)
        # If model doesn't exist, check for appropriate error
        else:
            self.assertIn('Model not found', json_data['error'])

if __name__ == '__main__':
    unittest.main()