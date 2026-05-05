import streamlit as st
import cv2
import time
import numpy as np
from datetime import datetime
import pandas as pd

# Services
from services.face_service import load_images, encode_faces
from services.attendance_service import mark
from services.email_service import send_email

# Database
from database.models import (
    create_tables,
    insert_employee,
    get_all_employees,
    get_today_attendance
)

# Config
from config.settings import (
    APP_NAME,
    ATTENDANCE_DURATION_SEC,
    FACE_MATCH_THRESHOLD,
    FRAME_SCALE
)

# =========================
# INIT DATABASE
# =========================
create_tables()

# =========================
# LOAD DATA
# =========================
images, ids, names = load_images()
encodings = encode_faces(images)

# Insert employees into DB (if not exists)
for emp_id, name in zip(ids, names):
    insert_employee(emp_id, name)

# =========================
# UI
# =========================
st.title(f"📷 {APP_NAME}")

st.write("Click below to start attendance (1 minute session)")

# =========================
# START ATTENDANCE
# =========================
if st.button("Start Attendance"):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera not available (Cloud does not support webcam)")
        st.stop()

    FRAME_WINDOW = st.image([])

    start_time = time.time()

    st.info("📷 Camera running...")

    while True:
        success, img = cap.read()
        if not success:
            st.error("Camera error")
            break

        # Resize for faster processing
        imgS = cv2.resize(img, (0, 0), None, FRAME_SCALE, FRAME_SCALE)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        import face_recognition
        faces = face_recognition.face_locations(imgS)
        encodes = face_recognition.face_encodings(imgS, faces)

        for encodeFace, faceLoc in zip(encodes, faces):
            matches = face_recognition.compare_faces(encodings, encodeFace)
            faceDis = face_recognition.face_distance(encodings, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < FACE_MATCH_THRESHOLD:
                emp_id = ids[matchIndex]
                name = names[matchIndex]
                mark(emp_id)
                label = f"{emp_id}-{name}"
                color = (0, 255, 0)
            else:
                label = "UNKNOWN"
                color = (0, 0, 255)

            # Scale back
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1 / FRAME_SCALE), int(x2 / FRAME_SCALE), int(y2 / FRAME_SCALE), int(x1 / FRAME_SCALE)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Stop after duration
        if time.time() - start_time > ATTENDANCE_DURATION_SEC:
            break

    cap.release()

    st.success("✅ Attendance Completed!")

    # =========================
    # GENERATE REPORT
    # =========================
    today = datetime.now().strftime("%Y-%m-%d")

    present_ids = get_today_attendance(today)
    employees = get_all_employees()

    present = [e for e in employees if e[0] in present_ids]
    absent = [e for e in employees if e[0] not in present_ids]

    # =========================
    # TABLE FORMAT
    # =========================
    data = []

    for emp_id, name in present:
        data.append({"EmpID": emp_id, "Name": name, "Status": "Present"})

    for emp_id, name in absent:
        data.append({"EmpID": emp_id, "Name": name, "Status": "Absent"})

    df_report = pd.DataFrame(data)

    st.write("### 📋 Attendance Table")
    st.dataframe(df_report, use_container_width=True)

    # =========================
    # CHART
    # =========================
    st.write("### 📊 Attendance Chart")

    if not df_report.empty:
        st.bar_chart(df_report["Status"].value_counts())

    # =========================
    # DOWNLOAD EXCEL
    # =========================
    file_name = f"Attendance_Report_{today}.xlsx"
    df_report.to_excel(file_name, index=False)

    with open(file_name, "rb") as f:
        st.download_button(
            label="📥 Download Excel Report",
            data=f,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # =========================
    # EMAIL
    # =========================
    try:
        send_email(present, absent)
        st.success("📧 Email sent to HR!")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")