from pathlib import Path
import streamlit as st

ROOT = Path(".")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def load_css():
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def card(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def empty_state(msg: str):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)

def kpi(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )