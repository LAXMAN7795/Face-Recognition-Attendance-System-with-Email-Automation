# config/settings.py
import os
from dotenv import load_dotenv

# 🔥 IMPORTANT LINE (YOU MISSED THIS)
load_dotenv()

# ---------- ENV / SECRETS ----------
def _get_secret(key: str, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


EMAIL_SENDER   = _get_secret("EMAIL_SENDER")
EMAIL_PASSWORD = _get_secret("EMAIL_PASSWORD")
EMAIL_RECEIVER = _get_secret("EMAIL_RECEIVER")

# ---------- APP SETTINGS ----------
APP_NAME = "Face Recognition Attendance System with Email Automation"

ATTENDANCE_DURATION_SEC = 60
FACE_MATCH_THRESHOLD = 0.5
FRAME_SCALE = 0.25

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

DB_PATH = os.path.join(BASE_DIR, "attendance.db")