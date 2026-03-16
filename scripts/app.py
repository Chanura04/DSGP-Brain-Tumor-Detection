import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import os



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


def predict_ct(image):
    """CT model prediction"""
    BASE_DIR = "C:\Users\chanu\Downloads\DenseNet_trained"
    dn_dir = os.path.join(BASE_DIR, "DenseNet_trained.keras")
    model = tf.keras.models.load_model(dn_dir)

    
    return "Tumor Detected" if np.random.random() > 0.5 else "No Tumor"


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
st.markdown("---")

# Create two columns for image upload
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 MRI Image Portal")
    mri_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"], key="mri")
    if mri_image:
        mri_img = Image.open(mri_image)

        st.image(mri_img, caption="Uploaded MRI Image", use_container_width=True)

        # segmented_img = segment_mri(mri_img)
        # st.image(segmented_img, caption="MRI Segmentation Output", use_container_width=True)


with col2:
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
        st.success("🟢 Healthy Scan - No Tumor Detected")
    else:
        with st.spinner("🔄 Processing images..."):
            st.markdown("---")
            st.header("📊 Results")

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

            

# Footer
st.markdown("---")
st.markdown("""
### 📝 Instructions:
1. Upload your MRI and/or CT images using the file uploaders above
2. Select the detection mode (MRI, CT, or Both)
3. Click the "Check for Tumor" button to start the analysis
4. View the detection results and segmentation output

**Note:** Replace the mock prediction functions with your actual trained models.
""")

#python -m streamlit run app.py