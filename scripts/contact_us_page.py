import streamlit as st


def contact_us_page():
    # 1. Page Config (Remove if already set in your main file)
    # st.set_page_config(page_title="Contact Us", page_icon="✉️", layout="wide")

    # 2. Enhanced CSS for Dark/Orange Theme
    st.markdown("""
    <style>


    /* Form Input Styling */
    div.stTextInput > div > div > input, 
    div.stTextArea > div > div > textarea {
        background-color: white !important;
        color: #ff8c00 !important;
        border: 1px solid rgba(255, 140, 0, 0.3) !important;
        border-radius: 10px !important;
    }

    div.stTextInput > div > div > input:focus, 
    div.stTextArea > div > div > textarea:focus {
        border: 1px solid #ff8c00 !important;
        box-shadow: 0 0 5px rgba(255, 140, 0, 0.5) !important;
    }

    /* Glassmorphism Card for Contact Details */
    .contact-info-card {
        background: rgba(255, 140, 0, 0.05);
        border: 1px solid rgba(255, 140, 0, 0.2);
        padding: 30px;
        border-radius: 20px;
        color: #e0e0e0;
    }

    .info-item {
        margin-bottom: 20px;
        font-size: 16px;
    }

    .info-label {
        color: #ff8c00;
        font-weight: bold;
        display: block;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1px;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #ff8c00 !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #e67e00 !important;
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.4) !important;
        transform: scale(1.02);
    }

    /* Section Titles */
    .contact-title {
        color: #ff8c00;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 10px;
        text-align: center;

    }
    </style>
    """, unsafe_allow_html=True)

    # 3. Content Layout
    st.markdown('<div class="contact-title">Get In Touch</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # Contact Form
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Full Name", placeholder="Enter your name")
            email = st.text_input("Email Address", placeholder="e.g. doctor@hospital.com")
            message = st.text_area("Your Message", placeholder="How can we help you?", height=150)

            submit = st.form_submit_button("SEND MESSAGE")

            if submit:
                if name and email and message:
                    st.success(f"Done! Thank you {name}, we've received your message.")
                    st.balloons()
                else:
                    st.error("Please fill in all the fields before sending.")

    with col2:
        # Sidebar-style Contact Info Card
        st.markdown(f"""
        <div class="contact-info-card">
            <h3 style="color: #ff8c00; margin-top:0;">Contact Details</h3>
            <div class="info-item">
                <span class="info-label">Email</span>
                <span style="color: black;">umordetect@gmail.com </span>
            </div>
            <div class="info-item">
                <span class="info-label"> Phone</span>
                <span style="color: black;">+94 11 22 4444 741</span>
            </div>
            <div class="info-item">
                <span class="info-label"> Location</span>
                <span style="color: black;"> Colombo, Sri Lanka </span>
            </div>
            <hr style="border: 0.5px solid rgba(255,140,0,0.2); margin: 20px 0;">
            <h3 style="color: #ff8c00; font-size: 18px;">Social Connect</h3>
            <div style="font-size: 20px;">
                <a href="https://linkedin.com" target="_blank" style="margin-right: 15px; font-size: 16px; color: black">🔗 LinkedIn</a><br>
                <a href="https://github.com/Chanura04/DSGP-Brain-Tumor-Detection" target="_blank" style="font-size: 16px; color: black"">🔗 GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
