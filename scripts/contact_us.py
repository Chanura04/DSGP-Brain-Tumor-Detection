import streamlit as st


def contactus():
    # Page config
    st.set_page_config(page_title="Contact Us", page_icon="", layout="wide")

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

    /* Input fields */
    div.stTextInput>div>input, div.stTextArea>div>textarea {
        background-color: #1a1a1a;
        color: #ff8c00;
        border: 1px solid #ff8c00;
    }

    /* Markdown links */
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
    st.title("Contact Us")
    st.markdown("Feel free to reach out to us!")

    # Contact Form
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")

        submit = st.form_submit_button("Send Message")

        if submit:
            if name and email and message:
                st.success("✅ Message sent successfully!")
                st.write("We will get back to you soon.")
            else:
                st.error("❌ Please fill in all fields.")

    # Additional Contact Info
    st.markdown("""
    ###  Our Details
    - 📧 Email: tumordetect@gmail.com
    - 📞 Phone: +94 -- -- ----
    - 📍 Location: Sri Lanka

    ### Follow Us
    - [LinkedIn](https://linkedin.com)
    - [GitHub](https://github.com/Chanura04/DSGP-Brain-Tumor-Detection)
    """)
