import streamlit as st
import cv2
import os
import time
import numpy as np
from datetime import datetime
from PIL import Image
import pandas as pd
import threading
import tempfile

# Import custom modules
from utils.face_verification import verify_identity
from utils.monitoring import check_person_in_frame, count_faces
from utils.logging_utils import log_event, generate_report

# Configure page
st.set_page_config(
    page_title="Exam Proctoring System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'reference_image' not in st.session_state:
    st.session_state.reference_image = None
if 'identity_verified' not in st.session_state:
    st.session_state.identity_verified = False
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

def main():
    # Page header
    st.title("AI-Based Exam Proctoring System")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.markdown("---")
        
        # Upload reference image
        st.subheader("Step 1: Upload Reference Image")
        uploaded_file = st.file_uploader("Upload a clear photo of your face", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save reference image to session state
            reference_img = Image.open(uploaded_file)
            st.session_state.reference_image = reference_img
            st.image(reference_img, caption="Reference Image", width=250)
            st.success("Reference image uploaded!")
        
        # Identity verification section
        st.markdown("---")
        st.subheader("Step 2: Identity Verification")
        
        verify_button = st.button("Verify Identity", disabled=(st.session_state.reference_image is None))
        
        # Monitoring controls section
        st.markdown("---")
        st.subheader("Step 3: Monitoring")
        
        start_button = st.button("Start Monitoring", 
                                disabled=(not st.session_state.identity_verified), 
                                type="primary")
        stop_button = st.button("Stop Monitoring", 
                               disabled=(not st.session_state.monitoring_active), 
                               type="secondary")
        
        st.markdown("---")
        download_button = st.download_button(
            label="Download Monitoring Report",
            data=generate_report(),
            file_name=f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            disabled=(len(st.session_state.log_data) == 0)
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    # Webcam feed
    with col1:
        st.subheader("Live Camera Feed")
        camera_placeholder = st.empty()
        
        # Status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            identity_status = st.empty()
        with status_col2:
            monitoring_status = st.empty()
        
        # Update status indicators
        if st.session_state.identity_verified:
            identity_status.success("Identity Verified")
        else:
            identity_status.error("Identity Not Verified")
            
        if st.session_state.monitoring_active:
            monitoring_status.success("Monitoring Active")
        else:
            monitoring_status.warning("Monitoring Inactive")
    
    # Event log
    with col2:
        st.subheader("Event Log")
        log_placeholder = st.empty()
        
        # Display logs
        if st.session_state.log_data:
            log_df = pd.DataFrame(st.session_state.log_data, 
                                 columns=["Timestamp", "Event", "Details"])
            log_placeholder.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            log_placeholder.info("No events recorded yet.")
    
    # Handle identity verification
    if verify_button:
        with st.spinner("Verifying identity..."):
            # Start webcam for verification
            st.session_state.webcam_active = True
            
            # Capture frame for verification
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access webcam! Please check your camera settings.")
                return
                
            ret, frame = cap.read()
            if ret:
                current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Display the captured image
                camera_placeholder.image(current_image, caption="Captured Image", use_container_width=True)
                
                # Verify identity
                verification_result, confidence = verify_identity(
                    np.array(st.session_state.reference_image), 
                    np.array(current_image)
                )
                
                if verification_result:
                    st.session_state.identity_verified = True
                    st.balloons()
                    st.success(f"Identity verified with {confidence:.2f}% confidence!")
                    log_event("Identity Verification", "Success")
                else:
                    st.session_state.identity_verified = False
                    st.error(f"Identity verification failed! Confidence: {confidence:.2f}%")
                    log_event("Identity Verification", "Failed")
            else:
                st.error("Failed to capture image from webcam!")
                
            cap.release()
    
    # Handle monitoring
    if start_button and not st.session_state.monitoring_active:
        st.session_state.monitoring_active = True
        st.session_state.webcam_active = True
        log_event("Monitoring", "Started")
        st.rerun()
        
    if stop_button and st.session_state.monitoring_active:
        st.session_state.monitoring_active = False
        st.session_state.webcam_active = False
        log_event("Monitoring", "Stopped")
        st.rerun()
    
    # Webcam monitoring loop
    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam! Please check your camera settings.")
            return
            
        try:
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if ret:
                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add monitoring status indicators to frame
                    if st.session_state.monitoring_active:
                        # Check if person is in frame
                        person_in_frame = check_person_in_frame(frame)
                        
                        # Count faces in frame
                        face_count = count_faces(frame)
                        
                        # Draw status on frame
                        if person_in_frame:
                            cv2.putText(frame_rgb, "Person detected", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame_rgb, "No person detected!", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if st.session_state.monitoring_active:
                                log_event("Violation", "No person in frame")
                        
                        cv2.putText(frame_rgb, f"Faces: {face_count}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 255, 0) if face_count == 1 else (0, 0, 255), 2)
                        
                        if face_count > 1 and st.session_state.monitoring_active:
                            log_event("Violation", f"Multiple persons detected ({face_count})")
                        elif face_count == 0 and st.session_state.monitoring_active:
                            log_event("Violation", "No face detected")
                    
                    # Display the frame
                    camera_placeholder.image(frame_rgb, caption="Live Feed", use_container_width=True)
                    
                    # Update log display if there are new logs
                    if st.session_state.log_data:
                        log_df = pd.DataFrame(st.session_state.log_data, 
                                             columns=["Timestamp", "Event", "Details"])
                        log_placeholder.dataframe(log_df, use_container_width=True, hide_index=True)
                    
                    # Slow down the loop slightly
                    time.sleep(0.1)
                
                if not st.session_state.webcam_active:
                    break
                    
        except Exception as e:
            st.error(f"Error in webcam processing: {str(e)}")
        finally:
            cap.release()
    
if __name__ == "__main__":
    main()