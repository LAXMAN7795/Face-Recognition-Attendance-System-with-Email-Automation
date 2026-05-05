import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from config.settings import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
import os


def send_email(present, absent):
    today = datetime.now().strftime('%Y-%m-%d')
    file_name = f"Attendance_Report_{today}.xlsx"

    # =========================
    # Create Email
    # =========================
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = f"Attendance Report - {today}"

    # =========================
    # Email Body
    # =========================
    body = "Attendance Report\n\n"

    body += "Present:\n"
    for p in present:
        body += f"{p[0]} - {p[1]}\n"

    body += "\nAbsent:\n"
    for a in absent:
        body += f"{a[0]} - {a[1]}\n"

    msg.attach(MIMEText(body, "plain"))

    # =========================
    # Attach Excel File
    # =========================
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())

        encoders.encode_base64(part)

        part.add_header(
            "Content-Disposition",
            f"attachment; filename={file_name}"
        )

        msg.attach(part)
    else:
        print("❌ Excel file not found:", file_name)

    # =========================
    # Send Email
    # =========================
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("✅ Email sent with attachment")