import streamlit as st
import time

def settings():

    st.markdown("<center><h3>Configurations</h3></center>", unsafe_allow_html=True)

    with st.form("settings_form"):
        frame_rate = st.number_input("Set Video Frame Rate", value=30)

        qc_cols = st.columns(2)
        with qc_cols[0]:
            videoqc = st.checkbox("Video", value=1)
            audioqc = st.checkbox("Audio", value=1)
        with qc_cols[1]:
            get_transcript = st.checkbox("📋 Get transcript after QC", value = st.session_state["get_transcript"])
        
        applybtn = st.form_submit_button("Apply")
    
    if applybtn:
        st.write(videoqc, audioqc, get_transcript, frame_rate)
        st.session_state["get_transcript"] = get_transcript
    return