import streamlit as st


# Dummy user credentials (replace with database later)
USER_CREDENTIALS = {
    "admin": "1234",
    "user": "pass"
}



def login():
    col1, col2, col3 = st.columns([1,3,1])

    with col2:

        st.title("Login")

        username = st.text_input("Username")    
        password = st.text_input("Password", type="password")

        login_btn = st.container()
       


        with login_btn:
            login_clicked = st.button("Login", use_container_width=True, key="log_btn")
            signup_clicked = st.button("Sign-Up",use_container_width=True)



        if login_clicked:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")


        if signup_clicked:
            st.success("proceed")



def logout():
    st.session_state.logged_in = False
    st.success("Logged out successfully")
    st.rerun()

def logged_in():
    st.title("Welcome to the App 🎉")
    st.write("You are logged in!")

    if st.button("Logout"):
        logout()

# App flow
def login_page():

    # Session state to track login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    
    if st.session_state.logged_in:
        logged_in()
    else:
        login()
