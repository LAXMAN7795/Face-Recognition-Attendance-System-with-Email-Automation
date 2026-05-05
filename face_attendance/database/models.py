from database.db import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id TEXT PRIMARY KEY,
        name TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT,
        date TEXT,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_employee(emp_id, name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT OR IGNORE INTO employees VALUES (?, ?)", (emp_id, name))

    conn.commit()
    conn.close()


def mark_attendance(emp_id, date, time):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM attendance WHERE emp_id=? AND date=?
    """, (emp_id, date))

    if cursor.fetchone() is None:
        cursor.execute("""
            INSERT INTO attendance(emp_id, date, time)
            VALUES (?, ?, ?)
        """, (emp_id, date, time))

    conn.commit()
    conn.close()


def get_all_employees():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM employees")
    data = cursor.fetchall()

    conn.close()
    return data


def get_today_attendance(date):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT emp_id FROM attendance WHERE date=?", (date,))
    data = cursor.fetchall()

    conn.close()
    return [d[0] for d in data]