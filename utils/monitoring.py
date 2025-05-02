import cv2
import numpy as np
import logging
from .face_verification import detect_face

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_person_in_frame(frame):
    """
    Check if a person is present in the frame using motion detection
    and face detection.
    
    Args:
        frame (numpy.ndarray): Current frame from webcam
        
    Returns:
        bool: True if a person is detected, False otherwise
    """
    try:
        # Method 1: Face detection
        faces = detect_face(frame)
        if len(faces) > 0:
            return True
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Initialize HOG descriptor for human detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect humans
        humans, _ = hog.detectMultiScale(
            resized_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        return len(humans) > 0
        
    except Exception as e:
        logger.error(f"Person detection error: {str(e)}")
        # Default to true in case of error to avoid false violations
        return True

def count_faces(frame):
    """
    Count the number of faces in the frame.
    
    Args:
        frame (numpy.ndarray): Current frame from webcam
        
    Returns:
        int: Number of faces detected
    """
    try:
        faces = detect_face(frame)
        return len(faces)
    except Exception as e:
        logger.error(f"Face counting error: {str(e)}")
        return 1  # Default to 1 in case of error to avoid false violations

def detect_abnormal_movement(frame, prev_frame):
    """
    Detect abnormal or sudden movements in the frame.
    
    Args:
        frame (numpy.ndarray): Current frame
        prev_frame (numpy.ndarray): Previous frame
        
    Returns:
        float: Movement intensity score (higher means more movement)
    """
    try:
        if prev_frame is None:
            return 0.0
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow (motion)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        # Calculate magnitude of flow
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate average magnitude
        movement_score = np.mean(magnitude)
        
        return movement_score
    except Exception as e:
        logger.error(f"Movement detection error: {str(e)}")
        return 0.0

def check_lighting_conditions(frame):
    """
    Check if lighting conditions are sufficient for monitoring.
    
    Args:
        frame (numpy.ndarray): Current frame
        
    Returns:
        tuple: (bool, float) - (is_lighting_good, brightness_value)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        brightness = np.mean(gray)
        
        # Define thresholds
        min_brightness = 40  # Minimum acceptable brightness
        max_brightness = 220  # Maximum acceptable brightness
        
        # Check if brightness is within acceptable range
        is_good = min_brightness <= brightness <= max_brightness
        
        return is_good, brightness
    except Exception as e:
        logger.error(f"Lighting check error: {str(e)}")
        return True, 128  # Default to good lighting in case of error

def detect_phone_or_object(frame):
    """
    Try to detect phones or unauthorized objects.
    This is a simplified implementation that can be expanded.
    
    Args:
        frame (numpy.ndarray): Current frame
        
    Returns:
        bool: True if suspicious object is detected
    """

    return False

def draw_monitoring_status(frame, person_in_frame, face_count, brightness_ok):
    """
    Draw monitoring status on the frame.
    
    Args:
        frame (numpy.ndarray): Current frame
        person_in_frame (bool): Whether person is detected
        face_count (int): Number of faces detected
        brightness_ok (bool): Whether lighting is good
        
    Returns:
        numpy.ndarray: Frame with status information
    """
    output_frame = frame.copy()
    
    # Draw status text
    cv2.putText(
        output_frame, 
        f"Person in frame: {'Yes' if person_in_frame else 'No'}", 
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
        (0, 255, 0) if person_in_frame else (0, 0, 255), 
        2
    )
    
    cv2.putText(
        output_frame, 
        f"Faces detected: {face_count}", 
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
        (0, 255, 0) if face_count == 1 else (0, 0, 255), 
        2
    )
    
    cv2.putText(
        output_frame, 
        f"Lighting: {'Good' if brightness_ok else 'Poor'}", 
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
        (0, 255, 0) if brightness_ok else (0, 255, 255), 
        2
    )
    
    # Draw status rectangle
    status_ok = person_in_frame and face_count == 1 and brightness_ok
    status_color = (0, 255, 0) if status_ok else (0, 0, 255)
    cv2.rectangle(output_frame, (5, 5), (300, 100), status_color, 2)
    
    return output_frame