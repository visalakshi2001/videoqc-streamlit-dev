import streamlit as st
import time

def settings():

    st.markdown("<center><h3>Configurations</h3></center>", unsafe_allow_html=True)

    with st.form("settings_form"):
        frame_rate = st.number_input("Set Video Frame Rate", value=30)

        videoqc = st.checkbox("Video")
        audioqc = st.checkbox("Audio")
        scriptqc = st.checkbox("Script")
        
        applybtn = st.form_submit_button("Apply")
    
    if applybtn:
        st.write(videoqc, audioqc, scriptqc, frame_rate)
    return