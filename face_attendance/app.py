import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText

# =========================
# Load ENV
# =========================
load_dotenv()
SENDER = os.getenv("EMAIL_SENDER")
PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER = os.getenv("EMAIL_RECEIVER")

# =========================
# Load Images
# =========================
path = 'images'
images = []
classNames = []
classIDs = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    if img is not None:
        images.append(img)
        file_name = os.path.splitext(file)[0]
        emp_id, name = file_name.split("_")
        classIDs.append(emp_id)
        classNames.append(name.upper())

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
    return encodeList

encodeListKnown = findEncodings(images)

# =========================
# Mark Attendance
# =========================
def markAttendance(emp_id, name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("EmpID,Name,Date,Time\n")

    with open("attendance.csv", "r+") as f:
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
# Load Employees
# =========================
def get_all_employees():
    employees = []
    with open("employees.txt", "r") as f:
        for line in f:
            emp_id, name = line.strip().split(",")
            employees.append((emp_id, name.upper()))
    return employees

# =========================
# Send Email
# =========================
def send_email(present, absent):
    subject = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"

    body = "Attendance Report\n\n"

    body += f"Present ({len(present)}):\n"
    for p in present:
        body += f"{p[0]} - {p[1]}\n"

    body += f"\nAbsent ({len(absent)}):\n"
    for a in absent:
        body += f"{a[0]} - {a[1]}\n"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER
    msg['To'] = RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER, PASSWORD)
        server.send_message(msg)

# =========================
# STREAMLIT UI
# =========================
st.title("📷 Face Recognition Attendance System")

if st.button("Start Attendance"):

    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    start_time = time.time()
    DURATION = 60

    st.info("Camera running...")

    while True:
        success, img = cap.read()
        if not success:
            st.error("Camera error")
            break

        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
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
                markAttendance(emp_id, name)
                label = f"{emp_id}-{name}"
                color = (0,255,0)
            else:
                label = "UNKNOWN"
                color = (0,0,255)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.putText(img,label,(x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img)

        if time.time() - start_time > DURATION:
            break

    cap.release()

    st.success("Attendance Completed!")

    # Generate Report
    df = pd.read_csv("attendance.csv")
    today = datetime.now().strftime('%Y-%m-%d')

    present_ids = df[df['Date'] == today]['EmpID'].tolist()
    present_ids = list(set(present_ids))

    employees = get_all_employees()

    present = [emp for emp in employees if emp[0] in present_ids]
    absent = [emp for emp in employees if emp[0] not in present_ids]

    st.write("### 📊 Attendance Summary")
    # =========================
    # Format Data for Table
    # =========================
    data = []

    for emp_id, name in present:
        data.append({"EmpID": emp_id, "Name": name, "Status": "Present"})

    for emp_id, name in absent:
        data.append({"EmpID": emp_id, "Name": name, "Status": "Absent"})

    df_report = pd.DataFrame(data)

    # =========================
    # Show Table
    # =========================
    st.write("### 📋 Attendance Table")
    st.dataframe(df_report, use_container_width=True)

    # =========================
    # Show Chart
    # =========================
    st.write("### 📊 Attendance Chart")

    status_counts = df_report["Status"].value_counts()

    st.bar_chart(status_counts)

    # =========================
    # Download Excel
    # =========================
    today = datetime.now().strftime('%Y-%m-%d')
    file_name = f"Attendance_Report_{today}.xlsx"

    df_report.to_excel(file_name, index=False)

    with open(file_name, "rb") as f:
        st.download_button(
            label="📥 Download Excel Report",
            data=f,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    send_email(present, absent)

    st.success("📧 Email Sent to HR!")