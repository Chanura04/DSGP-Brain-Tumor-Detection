import streamlit as st

from PIL import Image
import io
# HOME
# import streamlit as st
# from PIL import Image
# import io

from src.utils.image_utils import is_too_black, is_too_white, IMAGE_DISPLAY_SIZE
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS

from services.model_manager import head_detection, ct_tumor_detection, mri_tumor_classification, \
    tumor_segmentation, overlay_mask
from services.database_manager import generate_feedback_id, save_radiologist_data, save_text_report


from login_signup import login_router
from about_us import aboutus
from contact_us import contactus

# ---------------- GLOBAL CSS ----------------
def load_global_css():
    st.markdown("""
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px !important;
        width: 95% !important;
        margin: auto;
    }

    section[data-testid="stHorizontalBlock"] {
        position: sticky;   /* 🔥 important */
        top: 0;
        z-index: 999;
        background: rgba(15, 15, 20, 0.9);
        backdrop-filter: blur(12px);
        padding-top: 10px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }



    div.stButton > button {
        background: linear-gradient(145deg, #ffa733, #ff7a33) !important;
        border-radius: 15px !important;
        padding: 10px 20px !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        width: auto !important;
        min-width: 300px;
        display: block;
        margin: auto;
    }

    div.stButton > button:hover {
        background: linear-gradient(145deg, #ff8c00, #e65c00) !important;
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    div[data-testid="column"] {
        padding: 0 5px !important;
    }

    div.stButton {
        margin: 0 !important;
    }

    div.stButton > button {
        display: block !important;
    }

    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
                

    </style>
    """, unsafe_allow_html=True)



def profile_page():
    if 'user' not in st.session_state or st.session_state.user is None:
        st.warning("Please log in to view your profile.")
        if st.button("Go to Login"):
            st.session_state.page = "login" 
            st.rerun()
        return

    user = st.session_state.user


    st.markdown("""
        <style>
            .profile-card {
                background: #ffffff;
                padding: 25px 30px;
                border-radius: 15px;
                border: 1px solid #e6e6e6;
                box-shadow: 0px 3px 10px rgba(0,0,0,0.06);
                margin-bottom: 20px;
                font-family: 'Source Sans Pro', sans-serif;
            }
            .profile-title {
                font-size: 24px;
                font-weight: 700;
                color: #1f77b4;
                margin-bottom: 20px;
                border-bottom: 2px solid #f0f2f6;
                padding-bottom: 10px;
            }
            .profile-item {
                font-size: 18px;
                margin-bottom: 12px;
                display: flex;
                justify-content: list-item;
            }
            .profile-label {
                font-weight: 600;
                color: #444;
                width: 100px;
                display: inline-block;
            }
            .profile-value {
                color: #000;
                font-weight: 400;
            }
             .contact-title {
                color: #ff8c00;
                font-size: 39px;
                font-weight: 800;
                margin-bottom: 10px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

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

# NAVBAR
def navbar():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nav_button("Home", "home")

    with col2:
        nav_button("About", "about")

    with col3:
        nav_button("Contact", "contact")

    with col4:
        if st.session_state.logged_in:
            if st.button("Profile"):
                st.session_state.page = "profile"
                st.rerun()
        else:
            if st.button("Login"):
                st.session_state.page = "login"
                st.rerun()


def nav_button(label, page):
    if st.button(label, key=page):
        st.session_state.page = page
        st.rerun()



def home():
    error = False

    # 1. CSS for Portal Styling
    st.markdown("""
        <style>
        .portal-card {
            background: rgba(255, 140, 0, 0.05);
            border: 1px solid rgba(255, 140, 0, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
        }
        .portal-header {
            color: black;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }
        /* Style the file uploader label to be invisible/small since we have headers */
        .stFileUploader label {
            color: #ff8c00 !important;
            font-size: 14px !important;
        }
        /* Image Preview Styling */
        .img-container {
            border: 2px solid #ff8c00;
            border-radius: 10px;
            padding: 5px;
            background: #000;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # --- CT SCAN COLUMN ---
    with col1:
        st.markdown("""
            <div class="portal-card">
                <div class="portal-header">CT IMAGE PORTAL</div>
            </div>
        """, unsafe_allow_html=True)
        
        ct_image = st.file_uploader("Select CT Scan", type=list(VALID_IMAGE_EXTENSIONS), key="ct")
        
        if ct_image:
            ct_file_bytes = ct_image.getvalue()
            img = Image.open(io.BytesIO(ct_file_bytes)).resize(IMAGE_DISPLAY_SIZE)
            
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img, caption="Analyzable CT Scan", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.success("CT Scan Loaded Successfully")

    # --- MRI SCAN COLUMN ---
    with col2:
        st.markdown("""
            <div class="portal-card">
                <div class="portal-header"> MRI IMAGE PORTAL</div>
            </div>
        """, unsafe_allow_html=True)
        
        mri_image = st.file_uploader("Select MRI Scan", type=list(VALID_IMAGE_EXTENSIONS), key="mri")
        
        if mri_image:
            mri_file_bytes = mri_image.getvalue()
            img = Image.open(io.BytesIO(mri_file_bytes)).resize(IMAGE_DISPLAY_SIZE)
            
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img, caption="Analyzable MRI Scan", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.success("MRI Scan Loaded Successfully")

    st.markdown("<br><hr style='border: 1px solid rgba(255,140,0,0.2)'>", unsafe_allow_html=True)

    # ---------------- CHECK BUTTON ----------------

    col3, col4, col5 = st.columns(3)

    with col4:
        if st.button("🔬 Check for Tumor", width="stretch"):

            if not ct_image or not mri_image:
                st.error("❌ Please upload both MRI and CT images!")
                error = True

            elif is_too_black(ct_file_bytes):
                st.error("❌ Invalid CT Image !")
                error = True

            elif is_too_white(ct_file_bytes):
                st.error("❌ Invalid CT Image!")
                error = True

            elif is_too_black(mri_file_bytes):
                st.error("❌ Invalid MRI Image!")
                error = True

            elif is_too_white(mri_file_bytes):
                st.error("❌ Invalid MRI Image!")
                error = True

            if error:
                st.stop()

            ct_head_detection_result, ct_head_detection_confidence = head_detection(ct_file_bytes)
            # print(f"CT head detection confidence: {ct_head_detection_confidence}%")
            print (ct_head_detection_confidence)
            if ct_head_detection_result == 0:
                st.error("❌ Please upload a valid head top-view CT image!")
                st.stop()

            mri_head_detection_result, mri_head_detection_confidence = head_detection(mri_file_bytes)
            # print(f"MRI head detection confidence: {mri_head_detection_confidence}%")
            print (mri_head_detection_confidence)
            if mri_head_detection_result == 0:
                st.error("❌ Please upload a valid head top-view MRI image!")
                st.stop()

            with st.spinner("Processing images..."):
                ct_tumor_result, ct_tumor_probability = ct_tumor_detection(ct_file_bytes)
                # print(f"CT Tumor Probability: {ct_tumor_probability}")

                if ct_tumor_result == "No Tumor Detected":
                    st.session_state.ct_tumor_result = ct_tumor_result
                    st.session_state.results_ready = True
                else:
                    mri_tumor_class, mri_tumor_probability = mri_tumor_classification(mri_file_bytes)

                    # print(f"MRI Tumor Predicted Class: {mri_tumor_class}")
                    # print(f"MRI Tumor Probability: {mri_tumor_probability}")

                    segmented_image = tumor_segmentation(mri_file_bytes)
                    overlay_image = overlay_mask(mri_file_bytes, segmented_image)

                    # Store everything in session state
                    st.session_state.ct_tumor_result = ct_tumor_result
                    st.session_state.mri_tumor_class = mri_tumor_class
                    st.session_state.mri_tumor_probability = mri_tumor_probability
                    st.session_state.results_ready = True
                    st.session_state.segmented_image = segmented_image
                    st.session_state.overlay_image = overlay_image
                    st.session_state.feedback_id = generate_feedback_id()

                    # Reset radiologist fields for new scan
                    st.session_state.report_submitted = False
    col1 = st.columns(1)[0]   
    with col1:
        if st.session_state.results_ready:
            if st.session_state.ct_tumor_result == "No Tumor Detected":
                st.success("🟢 Healthy Scan - No Tumor Detected")
            else:
                st.markdown("""
                    <style>
                    .results-card-error {
                        background: rgba(255, 0, 0, 0.1);
                        border: 2px solid #ff4b4b;
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        margin: 20px 0;
                        animation: pulse-red 2s infinite;
                    }
                    .error-text {
                        color: #ff4b4b;
                        font-size: 20px;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    @keyframes pulse-red {
                        0% { box-shadow: 0 0 0px rgba(255, 75, 75, 0.2); }
                        50% { box-shadow: 0 0 20px rgba(255, 75, 75, 0.6); }
                        100% { box-shadow: 0 0 0px rgba(255, 75, 75, 0.2); }
                    }
                    </style>
                    
                    <div class="results-card-error">
                        <div class="error-text">🔴 Unhealthy Scan - Tumor Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                ct_tumor_result = st.session_state.ct_tumor_result
                mri_tumor_class = st.session_state.mri_tumor_class
                mri_tumor_probability = st.session_state.mri_tumor_probability
                segmented_image = st.session_state.segmented_image
                overlay_image = st.session_state.overlay_image
                feedback_id = st.session_state.feedback_id


                # 1. Custom CSS for Results Styling
                st.markdown("""
                    <style>
                    /* Section Headers */
                    .section-title {
                        color: #ff8c00;
                        font-size: 28px;
                        font-weight: 800;
                        text-align: center;
                        margin-top: 30px;
                        text-transform: uppercase;
                        letter-spacing: 2px;
                    }
                    /* Divider */
                    .orange-divider {
                        height: 2px;
                        background: linear-gradient(90deg, transparent, #ff8c00, transparent);
                        margin: 10px 0 30px 0;
                    }
                    /* Report Card */
                    .report-card {
                        background: rgba(255, 255, 255, 0.05);
                        border-left: 5px solid #ff8c00;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    /* Centered Status Alert */
                    .status-alert {
                        text-align: center;
                        padding: 15px;
                        border-radius: 10px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                # --- RESULTS SECTION ---
                st.markdown('<div class="section-title"><br><br>Analysis Results</div>', unsafe_allow_html=True)
                st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

                result_col1, result_col2 = st.columns(2, gap="large")

                with result_col1:
                    st.subheader("📊 CT Analysis")
                    # Centered Error Alert
                    st.markdown(f'<div class="status-alert" style="background: rgba(255,75,75,0.2); border: 1px solid #ff4b4b; color: #ff4b4b;">🔴 {ct_tumor_result}</div>', unsafe_allow_html=True)
                    st.image(segmented_image, caption="CT Segmentation Preview", use_container_width=True)

                with result_col2:
                    st.subheader("🔬 MRI Analysis")
                    # Centered Error Alert
                    st.markdown(f'<div class="status-alert" style="background: rgba(255,75,75,0.2); border: 1px solid #ff4b4b; color: #ff4b4b;">🔴 {mri_tumor_class}</div>', unsafe_allow_html=True)
                    st.image(overlay_image, caption="MRI with Tumor Overlay", use_container_width=True)


                # --- DIAGNOSTIC REPORT SECTION ---
                st.markdown('<div class="section-title"><br><br>Diagnostic Report</div>', unsafe_allow_html=True)
                st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

                report_col1, report_col2 = st.columns([2, 1], gap="medium")

                with report_col1:
                    st.markdown(f"""
                    <div class="report-card">
                        <h3 style="color: #ff8c00; margin-top:0;">Patient Imaging Summary</h3>
                        <p><b>CT Status:</b> {ct_tumor_result}</p>
                        <p><b>MRI Status:</b> {mri_tumor_class}</p>
                        <p><b>AI Confidence Score:</b> <span style="color:red; font-size:20px;">{mri_tumor_probability:.2f}%</span></p>
                        <hr style="border: 0.5px solid rgba(255,140,0,0.2)">
                        <p style="font-size: 13px; color: #aaa;">⚠️ <i>This is an AI-assisted preliminary analysis. Final diagnosis must be confirmed by a clinical professional.</i></p>
                    </div>
                    """, unsafe_allow_html=True)

                with report_col2:
                    st.image(segmented_image, caption="MRI Segmentation Preview", use_container_width=True)


                # --- RADIOLOGIST FEEDBACK ---
                st.markdown('<div class="section-title"><br><br>Radiologist Comments</div>', unsafe_allow_html=True)
                st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

                st.info(f"🆔 Feedback Session ID: {feedback_id}")

                with st.form("radiologist_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        rad_name = st.text_input("Radiologist Name *")
                        rad_phone = st.text_input("Phone Number *")
                    with c2:
                        rad_email = st.text_input("Email Address *")
                        
                    rad_comment = st.text_area("Clinical Notes / Comments *", height=100)
                    submit_report = st.form_submit_button("✅ Submit Radiologist Report")

                    if submit_report:
                        if not rad_name or not rad_phone or not rad_email or not rad_comment:
                            st.error("Please fill all required fields.")
                        else:
                            save_radiologist_data(feedback_id, rad_name, rad_phone, rad_email, rad_comment, mri_tumor_class, ct_tumor_result)
                            st.session_state.report_submitted = True
                            st.success("✅ Radiologist report saved successfully!")

                # --- DOWNLOAD ACTION ---
                if st.session_state.get('report_submitted', False):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 DOWNLOAD OFFICIAL PDF REPORT",
                        data=save_text_report(feedback_id, ct_tumor_result, mri_tumor_class, rad_name, rad_phone, rad_email, rad_comment),
                        file_name=f"{feedback_id}_tumor_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )



# ---------------- MAIN APP ----------------
def main_app():

    

    st.set_page_config(page_title="Tumor Detection", layout="wide")

    # ✅ LOAD CSS FIRST
    load_global_css()

    # SESSION INIT
    defaults = {
        "logged_in": False,
        "page": "home",
        "ct_tumor_result": None,
        "mri_tumor_class": None,
        "mri_tumor_probability": 0,
        "results_ready": False,
        "segmented_image": None,
        "overlay_image": None,
        "feedback_id": None,
        "report_submitted": False
    }

    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    # HEADER
    # st.markdown("<h1 style='text-align:center;'>MRI & CT Tumor Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

        .main-header {
            font-family: 'Orbitron', sans-serif;
            text-align: center;
            color: #ff8c00;
            font-size: 50px;
            font-weight: 700;
            padding: 20px;
            margin-bottom: 10px;
            /* Text Glow Effect */
            text-shadow: 0 0 10px rgba(255, 140, 0, 0.5), 
                         0 0 20px rgba(255, 140, 0, 0.2);
            letter-spacing: 3px;
        }

        .header-underline {
            height: 4px;
            width: 100px;
            background: linear-gradient(90deg, transparent, #ff8c00, transparent);
            margin: 0 auto 30px auto;
            border-radius: 2px;
        }
        
        /* Subtle animation for the title */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .header-container {
            animation: fadeIn 1.5s ease-out;
        }
        </style>
        
        <div class="header-container">
            <h1 class="main-header">MRI & CT TUMOR DETECTION</h1>
            <div class="header-underline"></div>
        </div>
    """, unsafe_allow_html=True)

    # LOGIN GATE
    if not st.session_state.logged_in:
        login_router()
        st.stop()

    # NAVBAR
    navbar()

    st.markdown("---")

    # ROUTING
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "about":
        aboutus()
    elif st.session_state.page == "contact":
        contactus()
    elif st.session_state.page == "profile":
        profile_page()


main_app()
