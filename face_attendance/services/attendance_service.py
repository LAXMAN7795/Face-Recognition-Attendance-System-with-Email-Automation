# services/attendance_service.py
from database.models import mark_attendance
from utils.helpers import now_date, now_time

def mark(emp_id):
    mark_attendance(emp_id, now_date(), now_time())