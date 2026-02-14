import sqlite3
import numpy as np

DB_NAME = "faces.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding BLOB
        )
    """)
    conn.commit()
    conn.close()

def insert_face(name, encoding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO faces (name, encoding) VALUES (?, ?)",
        (name, encoding.tobytes())
    )
    conn.commit()
    conn.close()

def get_faces():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    conn.close()

    names = []
    encodings = []

    for row in rows:
        names.append(row[0])
        encodings.append(np.frombuffer(row[1], dtype=np.float64))

    return names, encodings
