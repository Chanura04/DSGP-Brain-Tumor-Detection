from pathlib import Path

import streamlit as st

from about_us_page import about_us_page
from contact_us_page import contact_us_page
from home_page import home_page
from login_signup import login_router
from profile_page import profile_page

from services.model_manager import load_head_detection_model, load_ct_tumor_detection_model, \
    load_mri_tumor_classification_model, load_tumor_segmentation_model

BASE_DIR = Path(__file__).parent.resolve()
ABS_DIR = BASE_DIR / "assets"


@st.cache_data
def load_css(file_name):
    return (ABS_DIR / file_name).read_text(encoding="utf-8")


def nav_button(label, page):
    if st.button(label, key=page):
        st.session_state.page = page
        st.rerun()


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


def main_app():
    @st.cache_resource
    def head_detection():
        return load_head_detection_model()

    @st.cache_resource
    def ct_tumor_detection():
        return load_ct_tumor_detection_model()

    @st.cache_resource
    def mri_tumor_classification():
        return load_mri_tumor_classification_model()

    @st.cache_resource
    def tumor_segmentation():
        return load_tumor_segmentation_model()

    head_detection_model, index, device = head_detection()
    ct_tumor_detection_model = ct_tumor_detection()
    mri_tumor_classification_model, classes = mri_tumor_classification()
    tumor_segmentation_model = tumor_segmentation()

    st.set_page_config(page_title="Tumor Detection", layout="wide")

    st.markdown(f"<style>{load_css('style.css')}</style>", unsafe_allow_html=True)

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
    st.markdown("""
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
        home_page(head_detection_model, index, ct_tumor_detection_model, mri_tumor_classification_model, classes,
                  tumor_segmentation_model, device)
    elif st.session_state.page == "about":
        about_us_page()
    elif st.session_state.page == "contact":
        contact_us_page()
    elif st.session_state.page == "profile":
        profile_page()


main_app()
