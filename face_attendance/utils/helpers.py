# utils/helpers.py
from datetime import datetime

def now_date():
    return datetime.now().strftime("%Y-%m-%d")

def now_time():
    return datetime.now().strftime("%H:%M:%S")

def unique_list(items):
    """Remove duplicates while preserving order"""
    seen = set()
    result = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def parse_image_filename(filename: str):
    """
    Expected format: E101_laxman.jpg
    Returns (emp_id, NAME)
    """
    import os
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}")
    emp_id, emp_name = parts
    return emp_id, emp_name.upper()