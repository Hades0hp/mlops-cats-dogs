import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
HEALTH_URL = f"{API_BASE}/health"
PREDICT_URL = f"{API_BASE}/predict"

st.set_page_config(page_title="Cats vs Dogs", page_icon="🐾")
st.title("🐾 Cats vs Dogs – Prediction UI")

# ---- Health Check ----
with st.expander("API Status", expanded=True):
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        r.raise_for_status()
        st.success(f"API OK: {r.json()}")
    except Exception as e:
        st.error(f"API not reachable: {e}")

st.divider()

# ---- Upload Image ----
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if file:
    st.image(file, caption=f"Selected: {file.name}", use_container_width=True)

    if st.button("Predict"):
        try:
            files = {
                "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
            }
            res = requests.post(PREDICT_URL, files=files, timeout=30)
            res.raise_for_status()

            st.subheader("Prediction Result")
            st.json(res.json())

        except Exception as e:
            st.error(f"Prediction failed: {e}")