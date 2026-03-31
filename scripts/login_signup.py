import streamlit as st
from database import create_users_table
from auth import signup, login

create_users_table()


# ---------------- LOGIN PAGE ----------------
def login_page():
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("<h2 style='text-align:center;'>Login</h2>", unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            success, message, user = login(email, password)

            if success:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error(message)

        if st.button("Go to Sign Up"):
            st.session_state.auth_page = "signup"
            st.rerun()


# ---------------- SIGNUP PAGE ----------------
def signup_page():
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("<h2 style='text-align:center;'>Sign Up</h2>", unsafe_allow_html=True)

        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")

            submit = st.form_submit_button("Create Account")

        if submit:
            success, message = signup(name, email, password, confirm)

            if success:
                st.success(message)
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.error(message)

        if st.button("Back to Login"):
            st.session_state.auth_page = "login"
            st.rerun()


# ---------------- ROUTER ----------------
def login_router():
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"

    if st.session_state.auth_page == "login":
        login_page()
    else:
        signup_page()
