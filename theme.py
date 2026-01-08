# theme.py — Light Retail Theme (restored-style, no HTML header block)
import streamlit as st
from typing import Optional

CSS = """
<style>
/* === Global Page Styling === */
html, body, [class^="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
               system-ui, -system-ui, sans-serif !important;
}

/* Reduce default padding */
.reportview-container .main .block-container,
.main .block-container {
  padding-top: 6px !important;
  padding-bottom: 12px !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
}

/* App background */
.stApp {
  background-color: #fdfdfd;
  background-image: url("https://www.transparenttextures.com/patterns/fabric-of-squares.png");
  background-attachment: fixed;
  background-size: cover;
}

/* Hide footer + menu (optional, cleaner look) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* === Headings === */
h1 {
  font-size: 2.4rem;
  letter-spacing: -0.6px;
  font-weight: 800;
}
h2 {
  font-size: 1.4rem;
  letter-spacing: -0.3px;
  font-weight: 700;
}

/* === Sidebar (compact, card-like) === */
section[data-testid="stSidebar"] {
  background-color: #F3F4F6;
  border-radius: 12px;
  padding: 12px 12px 16px 12px;
  border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] .sidebar-content {
  padding-top: 0 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-size: 0.95rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #4B5563;
}

.sidebar-title {
  font-size: 0.8rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #6B7280;
  margin-bottom: 0.35rem;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > label {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
}

.stRadio [data-baseweb="radio"] {
  padding-top: 0.12rem;
  padding-bottom: 0.12rem;
}

/* === Cards & containers === */
[data-testid="stDataFrame"],
[class*="stMarkdown"] > p,
[class*="stMarkdown"] > div {
  border-radius: 12px;
}

.css-1x8cf1d, .css-1d391kg {
  border-radius: 12px !important;
}

/* === Buttons === */
.stButton > button {
  background-color: #111827 !important;
  color: white !important;
  border-radius: 10px !important;
  height: 44px !important;
  font-weight: 600 !important;
  transition: all 0.14s ease;
  border: none !important;
  padding-left: 18px !important;
  padding-right: 18px !important;
}
.stButton > button:hover {
  background-color: #F59E0B !important;
  color: black !important;
  transform: translateY(-1px);
}

/* Primary buttons (kind="primary") */
button[kind="primary"] {
  background: linear-gradient(135deg, #F97316, #EA580C) !important;
  color: white !important;
}

/* === Inputs === */
.stTextInput > div > div > input,
.stNumberInput input {
  background-color: #F9FAFB !important;
  color: #111827 !important;
  border-radius: 8px !important;
  border: 1px solid #D1D5DB !important;
}
.stSelectbox > div > div {
  background-color: #F9FAFB !important;
  border-radius: 8px !important;
  border: 1px solid #D1D5DB !important;
}

/* Tables / Dataframes */
[data-testid="stDataFrame"] {
  border-radius: 10px;
  border: 1px solid #E5E7EB;
  overflow: hidden;
}

/* Expander */
.streamlit-expanderHeader {
  font-size: 0.92rem;
}

/* Divider */
hr {
  border-color: #E5E7EB;
}

/* Scrollbar (subtle) */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-thumb {
  background: #D1D5DB;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #9CA3AF;
}
</style>
"""

def apply_modern_retail_theme(logo_path: Optional[str] = None) -> None:
    """
    Apply a light retail-style theme by injecting CSS.
    `logo_path` is accepted for compatibility with app.py but not used here.
    """
    st.markdown(CSS, unsafe_allow_html=True)
