# 📷 Face Recognition Attendance System with Email Automation

A real-time face recognition-based attendance system that automatically detects faces using a webcam, marks attendance in a database, generates reports, and sends them via email with an Excel attachment.

---

## 🚀 Features

* 🔍 Real-time face detection using HOG
* 🤖 Face recognition using dlib CNN (128D embeddings)
* 🧑‍💼 Unique employee identification using Employee ID
* 🗄️ Attendance stored in SQLite database
* 📊 Attendance summary (Present / Absent)
* 📈 Visualization using charts (Streamlit)
* 📄 Excel report generation
* 📧 Automated email with Excel attachment
* ⚡ Optimized performance using image resizing (16× faster)

---

## 🧠 System Workflow

```
Start Application
↓
Load Images (Dataset)
↓
Generate Face Encodings (128D vectors)
↓
Start Webcam
↓
Detect Face (HOG)
↓
Generate Encoding (CNN - dlib)
↓
Compare with Stored Encodings
↓
Mark Attendance (Database)
↓
Generate Report (Excel)
↓
Send Email (SMTP)
```

---

## 🏗️ Architecture Diagram

![Architecture](architecture.png)

## 🏗️ Project Structure

```
face_attendance/
│
├── app.py                      # Streamlit UI & main controller
│
├── config/
│   └── settings.py             # Configuration & environment variables
│
├── database/
│   ├── db.py                   # Database connection
│   └── models.py               # Database queries (CRUD operations)
│
├── services/
│   ├── face_service.py         # Face loading & encoding
│   ├── attendance_service.py   # Attendance logic
│   └── email_service.py        # Email sending logic
│
├── utils/
│   └── helpers.py              # Utility functions
│
├── images/                     # Dataset (EmpID_Name.jpg)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Technologies Used

* Python
* OpenCV (cv2)
* face_recognition (dlib-based CNN)
* SQLite
* Streamlit
* Pandas
* OpenPyXL
* SMTP (Gmail)
* Python-dotenv

---

## 🤖 Face Recognition Details

* Model: dlib ResNet-based CNN (v1)
* Output: 128-dimensional face embedding

```
f(face) ∈ ℝ¹²⁸
```

* Matching method: Euclidean distance

```
d = √Σ(xᵢ - yᵢ)²
```

* Threshold used: `0.5`

---

## ⚡ Performance Optimization

* Image resized to 0.25 scale
* Pixel reduction: **16×**
* Processing time: ~50 ms per frame
* Maintains real-time performance with minimal accuracy loss (~3%)

---

## 🗄️ Database Design

### Employees Table

| Column | Type               |
| ------ | ------------------ |
| emp_id | TEXT (Primary Key) |
| name   | TEXT               |

### Attendance Table

| Column | Type    |
| ------ | ------- |
| id     | INTEGER |
| emp_id | TEXT    |
| date   | TEXT    |
| time   | TEXT    |

* Prevents duplicate attendance using `(emp_id + date)`

---

## 📧 Email Automation

* Uses SMTP (Gmail)
* MIME format for attachments
* Excel file encoded using Base64

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/face-recognition-attendance-system.git
cd face-recognition-attendance-system
```

---

### 2. Create virtual environment

```
python -m venv face_env
face_env\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Setup `.env` file

Create a `.env` file in root:

```
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=receiver_email@gmail.com
```

---

### 5. Add dataset images

Format:

```
EmpID_Name.jpg
Example:
E101_LAXMAN.jpg
E102_RAHUL.jpg
```

---

### 6. Run the application

```
streamlit run app.py
```

---

## 📊 Output

* Live face recognition via webcam
* Attendance table (Present / Absent)
* Chart visualization
* Excel report download
* Email sent automatically with attachment

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Performance may drop with extreme angles
* Webcam not supported in cloud deployment (Streamlit Cloud)

---

## 🚀 Future Improvements

* Employee registration UI
* Liveness detection (anti-spoofing)
* Cloud deployment (image upload-based)
* Database upgrade (MySQL / MongoDB)
* Role-based access (Admin/HR)

---

## 🎯 Key Learning Outcomes

* Computer vision (face detection & recognition)
* Deep learning embeddings
* Database design & integration
* Real-time system optimization
* Email automation using SMTP & MIME

---

## 👨‍💻 Author

**Laxman Sannu Gouda**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
