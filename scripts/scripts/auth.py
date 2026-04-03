import bcrypt
from database import get_connection


def signup(radiologist_name, email, password, confirm_password):
    if not radiologist_name or not email or not password or not confirm_password:
        return False, "All fields are required."

    if password != confirm_password:
        return False, "Passwords do not match."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT radiologist_id FROM users WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return False, "This email is already registered. Please log in."

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    cursor.execute("""
        INSERT INTO users (radiologist_name, email, password)
        VALUES (?, ?, ?)
    """, (radiologist_name, email, hashed_password))

    conn.commit()
    conn.close()

    return True, "Account created successfully. Please log in."


def login(email, password):
    if not email or not password:
        return False, "Please enter both email and password.", None

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT radiologist_id, radiologist_name, email, password
        FROM users
        WHERE email = ?
    """, (email,))

    user = cursor.fetchone()
    conn.close()

    if not user:
        return False, "Login details did not match any account.", "not_found"

    radiologist_id, radiologist_name, user_email, hashed_password = user

    if bcrypt.checkpw(password.encode("utf-8"), hashed_password):
        return True, "Login successful.", {
            "radiologist_id": radiologist_id,
            "radiologist_name": radiologist_name,
            "email": user_email
        }

    return False, "Password is incorrect.", "wrong_password"