import streamlit as st
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import os

from src.utils.image_utils import is_too_black, is_too_white, IMAGE_DISPLAY_SIZE
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS
<<<<<<< HEAD

<<<<<<< HEAD

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def head_detection_mri(image):
    pass

def head_detection_ct(image):
    pass

def predict_mri(image):
    """ MRI model prediction"""
    
    return "Tumor Detected" if np.random.random() > 0.5 else "No Tumor"
=======
from services.model_manager import mri_head_detection, ct_head_detection, ct_tumor_detection, mri_tumor_classification, \
    tumor_segmentation, overlay_mask
from services.database_manager import generate_feedback_id, save_radiologist_data, save_text_report
>>>>>>> master
=======

from services.model_manager import mri_head_detection, ct_head_detection, ct_tumor_detection, mri_tumor_classification, \
    tumor_segmentation, overlay_mask
from services.database_manager import generate_feedback_id, save_radiologist_data, save_text_report
>>>>>>> master

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

<<<<<<< HEAD
<<<<<<< HEAD
def predict_ct(image):
    """CT model prediction"""
    BASE_DIR = "C:\Users\chanu\Downloads\DenseNet_trained"
    dn_dir = os.path.join(BASE_DIR, "DenseNet_trained.keras")
    model = tf.keras.models.load_model(dn_dir)

    
    return "Tumor Detected" if np.random.random() > 0.5 else "No Tumor"
=======
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
>>>>>>> master

error = False

<<<<<<< HEAD
def segment_mri(pil_image):
    """Mock segmentation model"""

    model = load_model("C:\\Projects Datasets\\model_details\\unet_6\\model_2.h5",
                   custom_objects={"dice_coef": dice_coef},
                   compile=False)
    
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize to model input size
    image_resized = cv2.resize(image, (256, 256))

    # Normalize to [0,1]
    image_norm = image_resized / 255.0

    # Add batch dimension
    input_tensor = np.expand_dims(image_norm, axis=0)
    pred_mask = model.predict(input_tensor)

    # Remove batch dimension
    pred_mask = pred_mask[0]

    # Threshold (convert probabilities to binary mask)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    return (binary_mask.squeeze() * 255).astype(np.uint8)

def overlay_mask(original_img, mask):
    original = np.array(original_img)

    # Make mask 3-channel
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 0] = mask * 255   # Red channel

    overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
    return overlay

# Streamlit App
st.set_page_config(page_title="MRI and CT Tumor Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🏥 MRI and CT Tumor Detection System</h1>",
    unsafe_allow_html=True
)
=======
# ---------------- UI ----------------
st.set_page_config(page_title="MRI and CT Tumor Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏥 MRI and CT Tumor Detection System</h1>", unsafe_allow_html=True)
>>>>>>> master
=======
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

error = False

# ---------------- UI ----------------
st.set_page_config(page_title="MRI and CT Tumor Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏥 MRI and CT Tumor Detection System</h1>", unsafe_allow_html=True)
>>>>>>> master
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
<<<<<<< HEAD
<<<<<<< HEAD
    st.subheader("📷 MRI Image Portal")
    mri_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"], key="mri")
    if mri_image:
        mri_img = Image.open(mri_image)

        st.image(mri_img, caption="Uploaded MRI Image", use_container_width=True)
=======
=======
>>>>>>> master
    st.subheader("📷 CT Image Portal")
    ct_image = st.file_uploader("Upload CT Image", type=list(VALID_IMAGE_EXTENSIONS), key="ct")
    if ct_image:
        ct_file_bytes = ct_image.getvalue()
        st.image(Image.open(io.BytesIO(ct_file_bytes)).resize(IMAGE_DISPLAY_SIZE), caption="Uploaded CT Image",
                 width="stretch")
<<<<<<< HEAD
>>>>>>> master

        # segmented_img = segment_mri(mri_img)
        # st.image(segmented_img, caption="MRI Segmentation Output", use_container_width=True)


with col2:
<<<<<<< HEAD
    st.subheader("📷 CT Image Portal")
    ct_image = st.file_uploader("Upload CT Image", type=["jpg", "jpeg", "png"], key="ct")
    
    if ct_image:
        ct_img = Image.open(ct_image)
        st.image(ct_img, caption="Uploaded CT Image", use_container_width=True)

st.markdown("---")



# Check button
if st.button("🔬 Check for Tumor", type="primary", use_container_width=True):

    # Validate inputs
    if not mri_image or not ct_image:
        st.error("❌ Please upload both MRI and CT images!")
        st.stop()
    
    is_mri_top_view_head_detected = head_detection_mri(mri_img)
    is_ct_top_view_head_detected = head_detection_ct(ct_img)

    if is_ct_top_view_head_detected=="NO" and is_mri_top_view_head_detected=="No":
        st.error("❌ Please upload valid head top view images! ")
        st.stop()
        

    ct_result = predict_ct(ct_img)
    
    if ct_result=="No Tumor":
=======
    st.subheader("📷 MRI Image Portal")
    mri_image = st.file_uploader("Upload MRI Image", type=list(VALID_IMAGE_EXTENSIONS), key="mri")
    if mri_image:
        mri_file_bytes = mri_image.getvalue()
        st.image(Image.open(io.BytesIO(mri_file_bytes)).resize(IMAGE_DISPLAY_SIZE), caption="Uploaded MRI Image",
                 width="stretch")

st.markdown("---")

# ---------------- CHECK BUTTON ----------------
if st.button("🔬 Check for Tumor", type="primary", width="stretch"):

    if not ct_image or not mri_image:
        st.error("❌ Please upload both MRI and CT images!")
        error = True

    if is_too_black(ct_file_bytes):
        st.error("❌ CT Image too dark!")
        error = True

    if is_too_white(ct_file_bytes):
        st.error("❌ CT Image too light!")
        error = True

    if is_too_black(mri_file_bytes):
        st.error("❌ MRI Image too dark!")
        error = True

    if is_too_white(mri_file_bytes):
        st.error("❌ MRI Image too light!")
        error = True

    if error:
        st.stop()

    ct_head_detection_result, ct_head_detection_confidence = ct_head_detection(ct_file_bytes)
    # print(f"CT head detection confidence: {ct_head_detection_confidence}%")

    if ct_head_detection_result == 0:
        st.error("❌ Please upload a valid head top-view CT image!")
        st.stop()

    mri_head_detection_result, mri_head_detection_confidence = mri_head_detection(mri_file_bytes)
    # print(f"MRI head detection confidence: {mri_head_detection_confidence}%")

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
>>>>>>> master
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
        st.header("📊 Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.subheader("🩻 CT Analysis")
            st.error(f"🔴 {ct_tumor_result}")
            st.markdown("---")
            st.image(segmented_image, caption="CT Segmentation Preview", width="stretch")

<<<<<<< HEAD
        # Results columns
        result_col1, result_col2 = st.columns(2)

        # ---------------- MRI PROCESSING ----------------
        with result_col1:
            st.subheader("🧠 MRI Analysis")

            mri_result = predict_mri(mri_img)

            if "Tumor" in mri_result and "No" not in mri_result:
                st.error(f"🔴 {mri_result}")
                mri_status = "Tumor Detected"
            else:
                st.success(f"🟢 {mri_result}")
                mri_status = "No Tumor Detected"

            # Segmentation
            segmented_img = segment_mri(mri_img)
            # st.image(segmented_img, caption="MRI Segmentation Output", use_container_width=True)
            overlay_img = overlay_mask(mri_img, segmented_img)
            st.image(overlay_img,
                caption="Tumor Segmentation Overlay",
                use_container_width=True)



        # ---------------- CT PROCESSING ----------------
        with result_col2:
            st.subheader("🩻 CT Analysis")

            # ct_result = predict_ct(ct_img)

            if "Tumor" in ct_result and "No" not in ct_result:
                st.error(f"🔴 {ct_result}")
                ct_status = "Tumor Detected"
            else:
                st.success(f"🟢 {ct_result}")
                ct_status = "No Tumor Detected"
           

            # ---------------- DIAGNOSTIC REPORT ----------------
            st.markdown("---")
            st.header("📝 Diagnostic Report")

            report_col1, report_col2 = st.columns([2, 1])

            with report_col1:
                st.markdown(f"""
                ### 🏥 Patient Imaging Summary

                **MRI Result:** {mri_result}  
                **CT Result:** {ct_result}  

                ---
                **Interpretation:**  
                - MRI indicates: {mri_result}  
                - CT indicates: {ct_result}  

                ⚠️ This is an AI-assisted preliminary analysis. Please consult a certified radiologist for confirmation.
                """)

            with report_col2:
                st.image(segmented_img, caption="MRI Segmentation Preview", use_container_width=True)

            # Downloadable text report
            report_text = f"""
            MRI AND CT TUMOR DETECTION REPORT
            ----------------------------------

            MRI Result: {mri_result}
            CT Result: {ct_result}

            This report was generated by an AI-assisted system.
            Consult a medical professional for final diagnosis.
            """

            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name="tumor_detection_report.txt",
                mime="text/plain",
                use_container_width=True
            )

            
=======
        with result_col2:
            st.subheader("🧠 MRI Analysis")
            st.error(f"🔴 {mri_tumor_class}")
            st.markdown("---")

            st.image(overlay_image, caption="MRI with Tumor Overlay", width="stretch")
>>>>>>> master

        st.markdown("---")
        st.header("📝 Diagnostic Report")

<<<<<<< HEAD
**Note:** Replace the mock prediction functions with your actual trained models.
""")

#python -m streamlit run app.py
=======
=======

with col2:
    st.subheader("📷 MRI Image Portal")
    mri_image = st.file_uploader("Upload MRI Image", type=list(VALID_IMAGE_EXTENSIONS), key="mri")
    if mri_image:
        mri_file_bytes = mri_image.getvalue()
        st.image(Image.open(io.BytesIO(mri_file_bytes)).resize(IMAGE_DISPLAY_SIZE), caption="Uploaded MRI Image",
                 width="stretch")

st.markdown("---")

# ---------------- CHECK BUTTON ----------------
if st.button("🔬 Check for Tumor", type="primary", width="stretch"):

    if not ct_image or not mri_image:
        st.error("❌ Please upload both MRI and CT images!")
        error = True

    if is_too_black(ct_file_bytes):
        st.error("❌ CT Image too dark!")
        error = True

    if is_too_white(ct_file_bytes):
        st.error("❌ CT Image too light!")
        error = True

    if is_too_black(mri_file_bytes):
        st.error("❌ MRI Image too dark!")
        error = True

    if is_too_white(mri_file_bytes):
        st.error("❌ MRI Image too light!")
        error = True

    if error:
        st.stop()

    ct_head_detection_result, ct_head_detection_confidence = ct_head_detection(ct_file_bytes)
    # print(f"CT head detection confidence: {ct_head_detection_confidence}%")

    if ct_head_detection_result == 0:
        st.error("❌ Please upload a valid head top-view CT image!")
        st.stop()

    mri_head_detection_result, mri_head_detection_confidence = mri_head_detection(mri_file_bytes)
    # print(f"MRI head detection confidence: {mri_head_detection_confidence}%")

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
        st.header("📊 Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.subheader("🩻 CT Analysis")
            st.error(f"🔴 {ct_tumor_result}")
            st.markdown("---")
            st.image(segmented_image, caption="CT Segmentation Preview", width="stretch")

        with result_col2:
            st.subheader("🧠 MRI Analysis")
            st.error(f"🔴 {mri_tumor_class}")
            st.markdown("---")

            st.image(overlay_image, caption="MRI with Tumor Overlay", width="stretch")

        st.markdown("---")
        st.header("📝 Diagnostic Report")

>>>>>>> master
        report_col1, report_col2 = st.columns([2, 1])
        with report_col1:
            st.markdown(f"""
            ### 🏥 Patient Imaging Summary
    
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
        st.header("👨‍⚕️ Radiologist Comments")
        st.info(f"🆔 Feedback ID: {feedback_id}")

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
                label="📥 Download Report",
                data=save_text_report(feedback_id, ct_tumor_result, mri_tumor_class, rad_name, rad_phone, rad_email,
                                      rad_comment),
                file_name=f"{feedback_id}_tumor_report.txt",
                mime="text/plain",
                width="stretch"
            )
            st.markdown("---")
<<<<<<< HEAD
>>>>>>> master
=======
>>>>>>> master
