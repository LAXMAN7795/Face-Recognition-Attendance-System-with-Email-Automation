import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import time
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# =========================
# Load ENV variables
# =========================
load_dotenv()
SENDER = os.getenv("EMAIL_SENDER")
PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER = os.getenv("EMAIL_RECEIVER")

# =========================
# Create CSV if not exists
# =========================
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("EmpID,Name,Date,Time\n")

# =========================
# Load Images + Extract ID
# =========================
path = 'images'
images = []
classNames = []
classIDs = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    if img is not None:
        images.append(img)

        # File format: E101_laxman.jpg
        file_name = os.path.splitext(file)[0]
        parts = file_name.split("_")

        if len(parts) == 2:
            emp_id, name = parts
            classIDs.append(emp_id)
            classNames.append(name.upper())
        else:
            print(f"⚠️ Skipping invalid file name: {file}")

# =========================
# Encode Faces
# =========================
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
        else:
            encodeList.append(None)
    return encodeList

# =========================
# Mark Attendance
# =========================
def markAttendance(emp_id, name):
    with open('attendance.csv', 'r+') as f:
        lines = f.readlines()
        today = datetime.now().strftime('%Y-%m-%d')

        for line in lines:
            entry = line.strip().split(',')
            if len(entry) >= 4:
                if entry[0] == emp_id and entry[2] == today:
                    return

        now = datetime.now()
        f.writelines(f"{emp_id},{name},{today},{now.strftime('%H:%M:%S')}\n")

# =========================
# Email Function
# =========================

def send_email_with_attachment(present, absent):
    today = datetime.now().strftime('%Y-%m-%d')
    file_name = f"Attendance_Report_{today}.xlsx"

    # =========================
    # Create Excel File
    # =========================
    data = []

    for emp_id, name in present:
        data.append([emp_id, name, "Present"])

    for emp_id, name in absent:
        data.append([emp_id, name, "Absent"])

    df = pd.DataFrame(data, columns=["EmpID", "Name", "Status"])
    df.to_excel(file_name, index=False)

    # =========================
    # Create Email
    # =========================
    msg = MIMEMultipart()
    msg['From'] = SENDER
    msg['To'] = RECEIVER
    msg['Subject'] = f"Attendance Report - {today}"

    body = "Please find attached the attendance report."
    msg.attach(MIMEText(body, 'plain'))

    # =========================
    # Attach Excel File
    # =========================
    with open(file_name, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())

    encoders.encode_base64(part)
    part.add_header(
        'Content-Disposition',
        f'attachment; filename={file_name}'
    )

    msg.attach(part)

    # =========================
    # Send Email
    # =========================
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER, PASSWORD)
        server.send_message(msg)

    print(f"✅ Email sent with Excel attachment: {file_name}")

# =========================
# Load Employee List
# =========================
def get_all_employees():
    employees = []
    with open("employees.txt", "r") as f:
        for line in f:
            emp_id, name = line.strip().split(",")
            employees.append((emp_id, name.upper()))
    return employees

# =========================
# MAIN EXECUTION
# =========================

encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

start_time = time.time()
DURATION = 60  # 1 minute

print("📷 Camera started... (1 minute attendance)")

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, faces)

    for encodeFace, faceLoc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex]
            emp_id = classIDs[matchIndex]
            color = (0, 255, 0)
            markAttendance(emp_id, name)
            display_text = f"{emp_id}-{name}"
        else:
            display_text = "UNKNOWN"
            color = (0, 0, 255)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, display_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Attendance System', img)

    if time.time() - start_time > DURATION:
        print("⏱ Time completed")
        break

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# Generate Report
# =========================
df = pd.read_csv("attendance.csv")
today = datetime.now().strftime('%Y-%m-%d')

present_ids = df[df['Date'] == today]['EmpID'].tolist()
present_ids = list(set(present_ids))

all_employees = get_all_employees()

present = [emp for emp in all_employees if emp[0] in present_ids]
absent = [emp for emp in all_employees if emp[0] not in present_ids]

print("\n📊 Final Report")
print("Present:", present)
print("Absent :", absent)

# =========================
# Send Email
# =========================
send_email_with_attachment(present, absent)