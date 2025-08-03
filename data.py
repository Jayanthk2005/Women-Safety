import sqlite3
from datetime import datetime

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('gender_detection.db')
    cursor = conn.cursor()
    
    # Create IncidentLog table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS IncidentLog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            type TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Function to insert incident data into the IncidentLog table
def log_incident(latitude, longitude, incident_type):
    conn = sqlite3.connect('gender_detection.db')
    cursor = conn.cursor()
    
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Insert the incident details into the database
    cursor.execute('''
        INSERT INTO IncidentLog (timestamp, latitude, longitude, type) 
        VALUES (?, ?, ?, ?)
    ''', (timestamp, latitude, longitude, incident_type))
    
    conn.commit()
    conn.close()
