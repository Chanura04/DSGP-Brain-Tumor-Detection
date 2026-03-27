import streamlit as st


def aboutus():
    # Page config
    st.set_page_config(page_title="About Us", page_icon="", layout="wide")

    # Custom CSS for dark theme with dark orange accents
    st.markdown("""
    <style>
    /* Page background */
    body {
        background-color: #000000;
        color: #ff8c00;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #ff8c00;
    }
    

    /* Markdown lists */
    ul {
        color: #ff8c00;
    }

    /* Links */
    a {
        color: #ff8c00;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page content
    st.title("About Us")

    st.markdown("""
    This system is developed to assist medical personnel in the efficient detection, Classification and Segmentation of brain tumors 
    using deep learning techniques.

    ### Our Goal
    To provide a fast, reliable, and accessible AI-based solution to support radiologists 
    in diagnosing brain tumors from MRI and CT scans.

    ### Technologies Used
    - Python
    - Streamlit
    - TensorFlow / PyTorch
    - Custom CNN Models
    - Medical Imaging Processing

    ### Our Team
    - Kabilash Arunasalam  
    - Chanura 
    - Inazaman Sheshan Careem
    - Lahiru

    ### Key Highlights
    - Automated tumor classification
    - Image validation using CNN
    - High accuracy model performance
    - User-friendly interface
    """)
