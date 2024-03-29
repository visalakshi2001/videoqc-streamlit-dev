import streamlit as st
import time

def settings():

    st.markdown("<center><h3>Configurations</h3></center>", unsafe_allow_html=True)

    with st.form("settings_form"):
        frame_rate = st.number_input("Set Video Frame Rate", value=30, help="Choose a value from 15, 30, 45; The more the value, the faster will be the execution")

        qc_cols = st.columns(2)
        with qc_cols[0]:
            # do_videoqc = st.checkbox("Video", value=1)
            do_audioqc = st.checkbox("Include AudioQC with VideoQC Bulk Processing", value=1)
        with qc_cols[1]:
            get_transcript = st.checkbox("📋 Get transcript after QC", value = st.session_state["get_transcript"])
        
        applybtn = st.form_submit_button("Apply")
    
    if applybtn:
        st.write(do_audioqc, get_transcript, frame_rate)
        st.session_state["save_transcript"] = get_transcript
        st.session_state["frame_rate"] = frame_rate
        st.session_state["do_audioqc"] = do_audioqc
    return