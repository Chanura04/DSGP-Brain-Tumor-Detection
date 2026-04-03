import streamlit as st


import streamlit as st

def aboutus():
    # 1. Page Config (Note: This must be the very first Streamlit command in your script)
    # If you call this function from a main file that already has set_page_config, remove this line.
    # st.set_page_config(page_title="About Us", page_icon="🧠", layout="wide")

    # 2. Advanced CSS for Dark Orange High-Tech Theme
    st.markdown("""
    <style>
    
    
    .main-title {
        color: #ff8c00;
        font-size: 45px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Glassmorphism Card Style */
    .about-card {
        background: rgba(255, 140, 0, 0.05);
        border: 1px solid rgba(255, 140, 0, 0.3);
        padding: 25px;
        border-radius: 15px;
        color: black;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .about-card:hover {
        border: 1px solid #ff8c00;
        transform: translateY(-5px);
        background: rgba(255, 140, 0, 0.08);
    }

    .section-header {
        color: #ff8c00;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }

    /* Tech Tags */
    .tech-tag {
        display: inline-block;
        background: rgba(255, 140, 0, 0.2);
        color: #ff8c00;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        font-weight: 600;
        border: 1px solid rgba(255, 140, 0, 0.4);
    }

    /* Team Member Names */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        color:black
    }
    .team-member {
        color: black;
        padding: 10px;
        border-left: 3px solid #ff8c00;
        background: rgba(255, 255, 255, 0.03);
    }

    </style>
    """, unsafe_allow_html=True)

    # 3. Content
    st.markdown('<div class="main-title">About Our Mission</div>', unsafe_allow_html=True)

    # Intro Section
    st.markdown("""
    <div class="about-card">
        <div class="section-header"> System Overview</div>
        <p>This state-of-the-art system is developed to assist medical personnel in the efficient <b>Detection, Classification, and Segmentation</b> 
        of brain tumors. By leveraging deep learning, we bridge the gap between complex medical imaging and rapid diagnostic insights.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="about-card" style="height: 370px;">
            <div class="section-header"> Our Goal</div>
            <p>To provide a fast, reliable, and accessible AI-based solution that supports radiologists 
            in diagnosing brain tumors from MRI and CT scans with maximum precision.</p>
            <div class="section-header" style="margin-top:20px;"> Key Highlights</div>
            <ul style="color:black; font-size: 15px;">
                <li>Automated tumor classification</li>
                <li>Image validation using CNN</li>
                <li>High accuracy model performance</li>
                <li>User-friendly interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Technologies Section
        tech_html = "".join([f'<span class="tech-tag">{t}</span>' for t in 
                             ["Python", "Streamlit", "TensorFlow", "PyTorch", "CNN Models", "OpenCV", "Medical AI"]])
        
        st.markdown(f"""
        <div class="about-card" style="height: 370px;">
            <div class="section-header"> Technologies Used</div>
            <div style="margin-top: 10px;">{tech_html}</div>
            <div class="section-header" style="margin-top: 30px;"> Our Team</div>
            <div class="team-grid">
                <div class="team-member">Kabilash Arunasalam</div>
                <div class="team-member">Chanura</div>
                <div class="team-member">Inazaman Sheshan Careem</div>
                <div class="team-member">Lahiru</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<p style='text-align: center; color: #666; font-size: 12px; margin-top: 50px;'>© 2026 Medical AI Tumor Detection Project</p>", unsafe_allow_html=True)


