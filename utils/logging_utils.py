import os
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(event_type, details):
    """
    Log monitoring events to the session state and disk.
    
    Args:
        event_type (str): Type of event (e.g., "Violation", "Warning", "Info")
        details (str): Details about the event
    """
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry
        log_entry = [timestamp, event_type, details]
        
        # Add to session state if it's a new event or a violation
        # This prevents flooding the log with repeated normal events
        if (event_type == "Violation" or 
            len(st.session_state.log_data) == 0 or 
            st.session_state.log_data[-1][1] != event_type or 
            st.session_state.log_data[-1][2] != details):
            
            st.session_state.log_data.append(log_entry)
            
            # Keep only the last 100 events in memory
            if len(st.session_state.log_data) > 100:
                st.session_state.log_data = st.session_state.log_data[-100:]
        
        # Write to disk log file
        log_to_file(timestamp, event_type, details)
        
    except Exception as e:
        logger.error(f"Logging error: {str(e)}")

def log_to_file(timestamp, event_type, details):
    """
    Write log entry to a CSV file.
    
    Args:
        timestamp (str): Timestamp string
        event_type (str): Type of event
        details (str): Event details
    """
    try:
        # Create log directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Log filename based on date
        log_date = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/events_{log_date}.csv"
        
        # Check if file exists
        file_exists = os.path.isfile(log_file)
        
        # Write to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Event", "Details"])
            writer.writerow([timestamp, event_type, details])
            
    except Exception as e:
        logger.error(f"File logging error: {str(e)}")

def generate_report():
    """
    Generate a CSV report from the current session logs.
    
    Returns:
        str: CSV string containing log data
    """
    try:
        if not st.session_state.log_data:
            return "Timestamp,Event,Details\n"
            
        # Create DataFrame from log data
        log_df = pd.DataFrame(st.session_state.log_data, 
                             columns=["Timestamp", "Event", "Details"])
        
        # Return CSV string
        return log_df.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return "Error generating report"

def get_violation_statistics():
    """
    Calculate statistics about violations.
    
    Returns:
        dict: Dictionary with violation statistics
    """
    try:
        if not st.session_state.log_data:
            return {
                "total_violations": 0,
                "no_person_violations": 0,
                "multiple_faces_violations": 0,
                "other_violations": 0
            }
            
        # Count violations by type
        violations = [log for log in st.session_state.log_data if log[1] == "Violation"]
        
        no_person_violations = len([v for v in violations if "No person" in v[2]])
        multiple_faces_violations = len([v for v in violations if "Multiple persons" in v[2]])
        other_violations = len(violations) - no_person_violations - multiple_faces_violations
        
        return {
            "total_violations": len(violations),
            "no_person_violations": no_person_violations,
            "multiple_faces_violations": multiple_faces_violations,
            "other_violations": other_violations
        }
        
    except Exception as e:
        logger.error(f"Statistics calculation error: {str(e)}")
        return {"total_violations": 0}