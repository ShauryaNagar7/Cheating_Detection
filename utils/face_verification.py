import numpy as np
import cv2
from deepface import DeepFace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_identity(reference_image, current_image, model_name='VGG-Face', distance_metric='cosine'):
    """
    Verify if the person in the current image matches the reference image.
    
    Args:
        reference_image (numpy.ndarray): Reference image array
        current_image (numpy.ndarray): Current image array from webcam
        model_name (str): Face recognition model to use (default: 'VGG-Face')
        distance_metric (str): Distance metric for verification (default: 'cosine')
    
    Returns:
        tuple: (verification_result, confidence_score)
    """
    try:
        # Use DeepFace for verification
        result = DeepFace.verify(
            img1_path=reference_image,
            img2_path=current_image,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=True
        )
        
        # Extract results
        verified = result.get('verified', False)
        distance = result.get('distance', 1.0)
        
        # Convert distance to confidence (cosine distance is between 0-1, lower is better)
        # For cosine: confidence = (1 - distance) * 100
        # For euclidean: confidence = max(0, (1 - distance/threshold)) * 100
        if distance_metric == 'cosine':
            confidence = (1 - distance) * 100
        else:
            # Default threshold for other metrics
            threshold = result.get('threshold', 0.6)
            confidence = max(0, (1 - distance/threshold)) * 100
        
        logger.info(f"Verification result: {verified}, Confidence: {confidence:.2f}%")
        return verified, confidence
        
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}")
        return False, 0.0

def extract_face_embeddings(image, model_name='VGG-Face'):
    """
    Extract face embeddings from an image.
    
    Args:
        image (numpy.ndarray): Image array
        model_name (str): Face recognition model to use
    
    Returns:
        numpy.ndarray: Face embedding vector
    """
    try:
        # Use DeepFace to extract embeddings
        embedding = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            enforce_detection=True
        )
        return embedding
    except Exception as e:
        logger.error(f"Face embedding extraction error: {str(e)}")
        return None

def detect_face(image):
    """
    Detect faces in an image and return the face coordinates.
    
    Args:
        image (numpy.ndarray): Image array
    
    Returns:
        list: List of detected face coordinates (x, y, w, h)
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        return []

def highlight_faces(image, faces):
    """
    Draw rectangles around detected faces.
    
    Args:
        image (numpy.ndarray): Image array
        faces (list): List of face coordinates (x, y, w, h)
    
    Returns:
        numpy.ndarray: Image with highlighted faces
    """
    image_copy = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image_copy