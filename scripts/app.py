import streamlit as st
from PIL import Image
import io

from about_us import aboutus
from contact_us import contactus
from login import login_page
from src.utils.image_utils import is_too_black, is_too_white, IMAGE_DISPLAY_SIZE
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS

from services.model_manager import mri_head_detection, ct_head_detection, ct_tumor_detection, mri_tumor_classification, \
    tumor_segmentation, overlay_mask
from services.database_manager import generate_feedback_id, save_radiologist_data, save_text_report


def main_app():
    st.markdown("""
        <style>

            div.stButton {
                display: flex;
                justify-content: center;
            }

            div.stButton > button {
                background-color: #1a1a1a;
                color: #ff8c00;
                border-radius: 15px;
                padding: 8px 20px;
                border: none;
                width: 100%;
            }


            div.stButton > button:hover {
                background-color: #ff8c00;
                color: black;
            }


        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* General body */
        body {
            background-color: #000000;
            color: #ff8c00;
            font-family: 'Arial', sans-serif;
        }

        /* Navbar container with equal spacing */
        .navbar {
            display: flex;
            justify-content: space-around;  /* equally space links */
            align-items: center;
            background-color: #1a1a1a;
            padding:10px 0;
            border-radius: 20px;
            margin-bottom: 30px;
        }

        /* Navbar links */
        .navbar a {
            color: #ff8c00;
            text-decoration: none;
            padding: 3px 20px;
            border-radius: 15px;
            transition: 0.3s;
            font-weight: bold;
        }

        /* Hover effect */
        .navbar a:hover {
            background-color: #ff8c00;
            color: #000000;
        }

        /* Active page highlight */
        .navbar a.active {
            background-color: #ff8c00;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

    defaults = {
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
        if key not in st.session_state:
            st.session_state[key] = val

    if "page" not in st.session_state:
        st.session_state.page = "home"

    # ---------------- UI ----------------

    st.set_page_config(page_title="MRI and CT Tumor Detection", layout="wide")
    st.markdown("<h1 style='text-align: center;'> MRI and CT Tumor Detection System</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nav_button("Home", "home")
    with col2:
        nav_button("About Us", "about")
    with col3:
        nav_button("Contact Us", "contact")
    with col4:
        nav_button("Login", "login")

    st.markdown("---")

    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "about":
        aboutus()
    elif st.session_state.page == "contact":
        contactus()
    elif st.session_state.page == "login":
        login_page()


def home():
    error = False
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 CT Image Portal")
        ct_image = st.file_uploader("Upload CT Image", type=list(VALID_IMAGE_EXTENSIONS), key="ct")
        if ct_image:
            ct_file_bytes = ct_image.getvalue()
            st.image(Image.open(io.BytesIO(ct_file_bytes)).resize(IMAGE_DISPLAY_SIZE), caption="Uploaded CT Image",
                     width="stretch")

    with col2:
        st.subheader("📷 MRI Image Portal")
        mri_image = st.file_uploader("Upload MRI Image", type=list(VALID_IMAGE_EXTENSIONS), key="mri")
        if mri_image:
            mri_file_bytes = mri_image.getvalue()
            st.image(Image.open(io.BytesIO(mri_file_bytes)).resize(IMAGE_DISPLAY_SIZE), caption="Uploaded MRI Image",
                     width="stretch")

    st.markdown("---")

    # ---------------- CHECK BUTTON ----------------

    col2, col3, col4, col5, col6 = st.columns(5)

    with col4:
        if st.button("🔬 Check for Tumor", width="stretch"):

            if not ct_image or not mri_image:
                st.error("❌ Please upload both MRI and CT images!")
                error = True

            elif is_too_black(ct_file_bytes):
                st.error("❌ CT Image too dark!")
                error = True

            elif is_too_white(ct_file_bytes):
                st.error("❌ CT Image too light!")
                error = True

            elif is_too_black(mri_file_bytes):
                st.error("❌ MRI Image too dark!")
                error = True

            elif is_too_white(mri_file_bytes):
                st.error("❌ MRI Image too light!")
                error = True

            if error:
                st.stop()

            ct_head_detection_result, ct_head_detection_confidence = ct_head_detection(ct_file_bytes)
            # print(f"CT head detection confidence: {ct_head_detection_confidence}%")
            print(ct_head_detection_confidence)
            if ct_head_detection_result == 0:
                st.error("❌ Please upload a valid head top-view CT image!")
                st.stop()

            mri_head_detection_result, mri_head_detection_confidence = mri_head_detection(mri_file_bytes)
            # print(f"MRI head detection confidence: {mri_head_detection_confidence}%")
            print(mri_head_detection_confidence)
            if mri_head_detection_result == 0:
                st.error("❌ Please upload a valid head top-view MRI image!")
                st.stop()

            with st.spinner("🔄 Processing images..."):
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

    if st.session_state.results_ready:
        if st.session_state.ct_tumor_result == "No Tumor Detected":
            st.success("🟢 Healthy Scan - No Tumor Detected")
        else:
            st.error("🔴 Unhealthy Scan - Tumor Detected")
            ct_tumor_result = st.session_state.ct_tumor_result
            mri_tumor_class = st.session_state.mri_tumor_class
            mri_tumor_probability = st.session_state.mri_tumor_probability
            segmented_image = st.session_state.segmented_image
            overlay_image = st.session_state.overlay_image
            feedback_id = st.session_state.feedback_id

            st.markdown("---")
            st.header("Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.subheader("CT Analysis")
                st.error(f"🔴 {ct_tumor_result}")
                st.markdown("---")
                st.image(segmented_image, caption="CT Segmentation Preview", width="stretch")

            with result_col2:
                st.subheader("MRI Analysis")
                st.error(f"🔴 {mri_tumor_class}")
                st.markdown("---")

                st.image(overlay_image, caption="MRI with Tumor Overlay", width="stretch")

            st.markdown("---")
            st.header("Diagnostic Report")

            report_col1, report_col2 = st.columns([2, 1])
            with report_col1:
                st.markdown(f"""
                ### Patient Imaging Summary

                **CT Result:**

                    -  Status: {ct_tumor_result} 


                **MRI Result:** 

                    -  Confidence: {mri_tumor_probability:.2f}%
                    -  Status: {mri_tumor_class}

                ---

                ⚠️ This is an AI-assisted preliminary analysis.
                """)
            with report_col2:
                st.image(segmented_image, caption="MRI Segmentation Preview", width="stretch")

            st.markdown("---")
            st.header("Radiologist Comments")
            st.info(f" Feedback ID: {feedback_id}")

            with st.form("radiologist_form"):
                rad_name = st.text_input("Radiologist Name *")
                rad_phone = st.text_input("Phone Number *")
                rad_email = st.text_input("Email Address *")
                rad_comment = st.text_area("Clinical Notes / Comments *")

                submit_report = st.form_submit_button("✅ Submit Radiologist Report")

                if submit_report:
                    if not rad_name or not rad_phone or not rad_email or not rad_comment:
                        st.error("Please fill all required fields.")
                    else:
                        save_radiologist_data(feedback_id, rad_name, rad_phone, rad_email, rad_comment, mri_tumor_class,
                                              ct_tumor_result)
                        st.session_state.report_submitted = True
                        st.success("✅ Radiologist report saved successfully!")

            st.markdown("---")

            if st.session_state.report_submitted:
                st.download_button(
                    label="Download Report",
                    data=save_text_report(feedback_id, ct_tumor_result, mri_tumor_class, rad_name, rad_phone, rad_email,
                                          rad_comment),
                    file_name=f"{feedback_id}_tumor_report.txt",
                    mime="text/plain",
                    width="stretch"
                )
                st.markdown("---")


def nav_button(label, page_name):
    if st.button(label, key=page_name, use_container_width=True):
        st.session_state.page = page_name
        st.rerun()

    if st.session_state.page == page_name:
        st.markdown(f"""
        <style>
        button[kind="secondary"][data-testid="stButton"]#{page_name} {{
            background-color: #ff8c00 !important;
            color: black !important;
        }}
        </style>
        """, unsafe_allow_html=True)


main_app()