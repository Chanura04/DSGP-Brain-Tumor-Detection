import streamlit as st


def profile_page():
    if 'user' not in st.session_state or st.session_state.user is None:
        st.warning("Please log in to view your profile.")
        if st.button("Go to Login"):
            st.session_state.page = "login"
            st.rerun()
        return

    user = st.session_state.user

    st.markdown('<div class="contact-title">Profile</div>', unsafe_allow_html=True)

    html_code = f"""
    <div class="profile-card">
        <div class="profile-title">Welcome, {user['radiologist_name']}</div>
        <div class="profile-item">
            <span class="profile-label">ID:</span>
            <span class="profile-value">{user['radiologist_id']}</span>
        </div>
        <div class="profile-item">
            <span class="profile-label">Email:</span>
            <span class="profile-value">{user['email']}</span>
        </div>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)

    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.success("Logged out successfully!")
        st.rerun()
