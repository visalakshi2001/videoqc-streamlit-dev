from glob import glob
import streamlit as st
import io
import os
from datetime import datetime
import time
import pandas as pd
from stqdm import stqdm
import tempfile
from videoparser import extract_frames_main
from tests import detect_logo_position, predict_ethnicity_from_image, get_font_type, predict_accent
from cache_versions.singleapp import run_video_qc_tests

def batch_processing():
    
    st.markdown("<center><h3>Batch Video Processing</h3></center>", unsafe_allow_html=True)
    
    targets = []

    folder = st.session_state["project_path"]
    folder = os.path.join(folder, "02_Animation")
    with st.expander("Detected Topics"):
        for video in os.listdir(folder):
            st.text(video)
    with st.expander("Detected Videos"):
        for topic in os.listdir(folder):
            name = topic
            topic = os.path.join(folder, topic)
            targets.append(topic)
            st.text(f"{name} : {os.listdir(topic)}")

    st.write(f"Total topics to process: {len(targets)}")
    total_t = st.empty()

    for topic in stqdm(targets):
        time.sleep(2)
        for video in stqdm(glob(topic + "/*.mp4"), desc=f"{video} under analysis..."):
            st.write(video)
            time.sleep(5)

    data = pd.read_excel("D:\Drive_A\TMLC\All around VideoQC\Development\qcreport.xlsx")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            data.to_excel(writer, sheet_name='QCSheet1')

            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

            st.download_button(
            label="Download Excel QCSheet",
            data=buffer,
            file_name="qcreport.xlsx",
            mime="application/vnd.ms-excel"
            )


