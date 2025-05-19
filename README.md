# TumorDetect Pro

An advanced diagnostic tool that utilizes a sophisticated CNN model to classify brain MRI images into distinct tumor categories.

## Overview

TumorDetect Pro is a web-based application designed to assist neurologists and radiologists in the diagnostic process by providing intuitive visualizations and detailed classification reports for brain MRI scans. The application uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories:

1. **No Tumor** - Normal brain tissue without abnormal growth
2. **Glioma** - Tumors that originate in the glial cells of the brain
3. **Meningioma** - Tumors that arise from the meninges (the membranes that surround the brain and spinal cord)
4. **Pituitary** - Tumors that develop in the pituitary gland

## Features

- **Upload and Analysis**: Upload brain MRI scans in common image formats (JPG, PNG)
- **Real-time Classification**: Instant classification of uploaded images using a pre-trained CNN model
- **Visualization**: Visual representation of the classification results with probability distribution
- **Detailed Reports**: Comprehensive analysis with confidence scores for each tumor category
- **User-friendly Interface**: Intuitive web interface designed for medical professionals

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Machine Learning**: TensorFlow/Keras CNN model
- **Visualization**: Matplotlib for generating visual reports

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/KOSASIH/tumordetect-pro.git
cd tumordetect-pro
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the application in your web browser at:
```
http://localhost:8080
```

## Usage

1. Open the application in your web browser
2. Upload a brain MRI scan image (JPG or PNG format)
3. Click "Analyze Image" to process the scan
4. View the classification results, including:
   - Tumor type (if detected)
   - Confidence score
   - Probability distribution across all categories
   - Visual representation of the results

## Model Training

The CNN model used in TumorDetect Pro was trained on a dataset of brain MRI scans with labeled tumor categories. The model architecture consists of multiple convolutional layers followed by max pooling, flattening, and dense layers.

## Disclaimer

TumorDetect Pro is designed for educational and research purposes only. It should not be used as the sole basis for medical diagnosis. Always consult with qualified healthcare professionals for proper diagnosis and treatment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Open-source machine learning libraries
- Medical professionals who provided guidance and feedback