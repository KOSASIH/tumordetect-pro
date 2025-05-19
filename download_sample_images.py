import os
import urllib.request
import zipfile
import shutil

# Create sample images directory
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_images')
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Sample MRI images URLs (these are placeholder URLs - replace with actual URLs)
SAMPLE_IMAGES = [
    {
        "url": "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/raw/master/data/Training/glioma_tumor/image(1).jpg",
        "filename": "glioma_sample.jpg",
        "type": "Glioma"
    },
    {
        "url": "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/raw/master/data/Training/meningioma_tumor/image(1).jpg",
        "filename": "meningioma_sample.jpg",
        "type": "Meningioma"
    },
    {
        "url": "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/raw/master/data/Training/pituitary_tumor/image(1).jpg",
        "filename": "pituitary_sample.jpg",
        "type": "Pituitary"
    },
    {
        "url": "https://github.com/sartajbhuvaji/brain-tumor-classification-dataset/raw/master/data/Training/no_tumor/image(1).jpg",
        "filename": "no_tumor_sample.jpg",
        "type": "No Tumor"
    }
]

def download_sample_images():
    """Download sample MRI images for testing"""
    print("Downloading sample MRI images...")
    
    for image in SAMPLE_IMAGES:
        try:
            target_path = os.path.join(SAMPLE_DIR, image["filename"])
            print(f"Downloading {image['type']} sample to {target_path}...")
            
            # Download the image
            urllib.request.urlretrieve(image["url"], target_path)
            print(f"Downloaded {image['filename']}")
            
        except Exception as e:
            print(f"Error downloading {image['filename']}: {str(e)}")
    
    print(f"Sample images downloaded to {SAMPLE_DIR}")

if __name__ == "__main__":
    download_sample_images()
    print("Done!")