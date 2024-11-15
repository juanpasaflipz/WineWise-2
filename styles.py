import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            color: #722F37;
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            margin-bottom: 2rem;
        }
        .wine-card {
            background-color: #FFF;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .similarity-score {
            color: #722F37;
            font-weight: bold;
            font-size: 1.2rem;
        }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
