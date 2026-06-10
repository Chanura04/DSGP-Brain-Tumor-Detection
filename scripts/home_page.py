import io

import streamlit as st
from PIL import Image
from pydantic import ValidationError

from services.database_manager import save_radiologist_data, generate_feedback_id, save_text_report
from services.input_validator import RadiologistReportValidator
from services.inference_engine import detect_head, detect_tumor, classify_tumor, segment_tumor, mask_overlay
from src.utils.image_utils import is_too_black, is_too_white, IMAGE_DISPLAY_SIZE
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS


def ct_portal(col):
    with col:
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

            return ct_image, ct_file_bytes

        return ct_image, None


def mri_portal(col):
    with col:
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

            return mri_image, mri_file_bytes

        return mri_image, None


def image_validation(ct_image, mri_image, ct_file_bytes, mri_file_bytes):
    if not ct_image or not mri_image:
        st.error("❌ Please upload both MRI and CT images!")
        st.stop()

    elif is_too_black(ct_file_bytes):
        st.error("❌ Invalid CT Image !")
        st.stop()

    elif is_too_white(ct_file_bytes):
        st.error("❌ Invalid CT Image!")
        st.stop()

    elif is_too_black(mri_file_bytes):
        st.error("❌ Invalid MRI Image!")
        st.stop()

    elif is_too_white(mri_file_bytes):
        st.error("❌ Invalid MRI Image!")
        st.stop()


def head_detection_validation(head_detection_model, ct_file_bytes, mri_file_bytes, index, device):
    ct_head_detection_result = detect_head(head_detection_model, ct_file_bytes, index, device)

    if ct_head_detection_result == 0:
        st.error("❌ Please upload a valid head top-view CT image!")
        st.stop()

    mri_head_detection_result = detect_head(head_detection_model, mri_file_bytes, index, device)
    if mri_head_detection_result == 0:
        st.error("❌ Please upload a valid head top-view MRI image!")
        st.stop()


def process_images(ct_tumor_detection_model, mri_tumor_classification_model, tumor_segmentation_model, ct_file_bytes,
                   mri_file_bytes, classes, device):
    with st.spinner("Processing images..."):
        ct_tumor_result = detect_tumor(ct_tumor_detection_model, ct_file_bytes)

        if ct_tumor_result == "No Tumor Detected":
            st.session_state.ct_tumor_result = ct_tumor_result
            st.session_state.results_ready = True
        else:
            mri_tumor_class, mri_tumor_probability = classify_tumor(mri_tumor_classification_model, mri_file_bytes,
                                                                    classes, device)

            segmented_image = segment_tumor(tumor_segmentation_model, mri_file_bytes)
            overlay_image = mask_overlay(mri_file_bytes, segmented_image)

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


def results_analysis():
    st.markdown('<div class="section-title"><br><br>Analysis Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    result_col1, result_col2 = st.columns(2, gap="large")

    with result_col1:
        st.subheader("📊 CT Analysis")
        # Centered Error Alert
        st.markdown(
            f'<div class="status-alert" style="background: rgba(255,75,75,0.2); border: 1px solid #ff4b4b; color: #ff4b4b;">🔴 {st.session_state.ct_tumor_result}</div>',
            unsafe_allow_html=True)
        st.image(st.session_state.segmented_image, caption="CT Segmentation Preview", use_container_width=True)

    with result_col2:
        st.subheader("🔬 MRI Analysis")
        # Centered Error Alert
        st.markdown(
            f'<div class="status-alert" style="background: rgba(255,75,75,0.2); border: 1px solid #ff4b4b; color: #ff4b4b;">🔴 {st.session_state.mri_tumor_class}</div>',
            unsafe_allow_html=True)
        st.image(st.session_state.overlay_image, caption="MRI with Tumor Overlay", use_container_width=True)

    # --- DIAGNOSTIC REPORT SECTION ---
    st.markdown('<div class="section-title"><br><br>Diagnostic Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    report_col1, report_col2 = st.columns([2, 1], gap="medium")

    with report_col1:
        st.markdown(f"""
        <div class="report-card">
            <h3 style="color: #ff8c00; margin-top:0;">Patient Imaging Summary</h3>
            <p><b>CT Status:</b> {st.session_state.ct_tumor_result}</p>
            <p><b>MRI Status:</b> {st.session_state.mri_tumor_class}</p>
            <p><b>AI Confidence Score:</b> <span style="color:red; font-size:20px;">{st.session_state.mri_tumor_probability:.2f}%</span></p>
            <hr style="border: 0.5px solid rgba(255,140,0,0.2)">
            <p style="font-size: 13px; color: #aaa;">⚠️ <i>This is an AI-assisted preliminary analysis. Final diagnosis must be confirmed by a clinical professional.</i></p>
        </div>
        """, unsafe_allow_html=True)

    with report_col2:
        st.image(st.session_state.segmented_image, caption="MRI Segmentation Preview", use_container_width=True)


def radiologist_feedback():
    st.markdown('<div class="section-title"><br><br>Radiologist Comments</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    st.info(f"🆔 Feedback Session ID: {st.session_state.feedback_id}")

    with st.form("radiologist_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.rad_name = st.text_input("Radiologist Name *")
            st.session_state.rad_phone = st.text_input("Phone Number *")
        with c2:
            st.session_state.rad_email = st.text_input("Email Address *")

        st.session_state.rad_comment = st.text_area("Clinical Notes / Comments *", height=100)
        submit_report = st.form_submit_button("✅ Submit Radiologist Report")

        if submit_report:
            try:
                report = RadiologistReportValidator(name=st.session_state.rad_name, phone=st.session_state.rad_phone,
                                                    email=st.session_state.rad_email,
                                                    comments=st.session_state.rad_comment)
            except ValidationError:
                st.error("Invalid input. Please check your details.")
                st.stop()
            else:
                save_radiologist_data(st.session_state.feedback_id, report.name, report.phone, report.email,
                                      report.comments,
                                      st.session_state.mri_tumor_class, st.session_state.ct_tumor_result)
                st.session_state.report_submitted = True
                st.success("✅ Radiologist report saved successfully!")


def download_report():
    if st.session_state.get('report_submitted', False):
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 DOWNLOAD OFFICIAL PDF REPORT",
            data=save_text_report(st.session_state.feedback_id, st.session_state.ct_tumor_result,
                                  st.session_state.mri_tumor_class, st.session_state.rad_name,
                                  st.session_state.rad_phone, st.session_state.rad_email,
                                  st.session_state.rad_comment),
            file_name=f"{st.session_state.feedback_id}_tumor_report.txt",
            mime="text/plain",
            use_container_width=True
        )


def home_page(head_detection_model, index, ct_tumor_detection_model, mri_tumor_classification_model, classes,
                  tumor_segmentation_model, device):
    col1, col2 = st.columns(2, gap="large")

    ct_image, ct_file_bytes = ct_portal(col1)
    mri_image, mri_file_bytes = mri_portal(col2)

    # ---------------- CHECK TUMOR ----------------
    col3, col4, col5 = st.columns(3)

    with col4:
        if st.button("🔬 Check for Tumor", width="stretch"):
            image_validation(ct_image, mri_image, ct_file_bytes, mri_file_bytes)
            head_detection_validation(head_detection_model, ct_file_bytes, mri_file_bytes, index, device)
            process_images(ct_tumor_detection_model, mri_tumor_classification_model, tumor_segmentation_model,
                           ct_file_bytes, mri_file_bytes, classes, device)

    col1 = st.columns(1)[0]
    with col1:
        if st.session_state.results_ready:
            if st.session_state.ct_tumor_result == "No Tumor Detected":
                st.success("🟢 Healthy Scan - No Tumor Detected")
            else:
                st.markdown("""
                        <div class="results-card-error">
                            <div class="error-text">🔴 Unhealthy Scan - Tumor Detected</div>
                        </div>
                        """, unsafe_allow_html=True)

                results_analysis()
                radiologist_feedback()
                download_report()
