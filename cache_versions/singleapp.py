from glob import glob
import streamlit as st
import io
import os
from datetime import datetime
import pandas as pd
import tempfile
from videoparser import extract_frames_main
from tests import (detect_logo_position, predict_ethnicity_from_image, 
                    get_ost_font_type, predict_accent, get_audio_patchwork,
                    get_audio_stats)



def single_processing():
    
    st.markdown("<center><h3>Single Video Processing</h3></center>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a Video file", type=["mp4"])



    if uploaded is not None:

        with st.columns([2,2])[0]:
            st.video(uploaded)
        
        # read uploaded file
        st.session_state["vid_name"] = uploaded.name
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        

        data = run_video_qc_test_single(tfile.name)
        
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
        


def insert_data(dataframe,  duration,  qc_pnt, type, remarks=""):

    """
    ["Session No", "Topic Name", "Duration", "Developed by", "Reported by",
                                    "Date", "QCPoint", "Type", "remarks", "Path"]
    """
    
    project_name = st.session_state["project_path"].split("/")[-1].split("\\")[-1]
    qc_dev= st.session_state["qc_dev_mem"]
    qc_mem= st.session_state["qc_mem_name"]
    topic_name= st.session_state["vid_name"]
    path = os.path.join(st.session_state["project_path"], r"02_Animation", st.session_state["session_name"])

    print(path)
    dataframe.loc[len(dataframe)] = [project_name, topic_name, duration, qc_dev, qc_mem,
                                     datetime.today().strftime("%d-%m-%Y"), qc_pnt, type, remarks, path]
    
    return dataframe

# https://docs.streamlit.io/library/advanced-features/st.cache
@st.cache(suppress_st_warning=True)
def run_video_qc_test_single(input_video):
    

    result, complexity_scale, stat_dict = get_audio_stats(input_video)


    
    # extract 
    with st.spinner("🧩 Parsing video frames..."):
        extract_frames_main(input_video)
    
    st.write("Parsing complete, running test cases:")

    data = pd.DataFrame(columns = ["Session No", "Topic Name", "Duration", "Developed by", "Reported by",
                                    "Date", "QCPoint", "Type", "remarks", "Path"])
    
    time_range = get_audio_patchwork(input_video)
    
    for start, end in zip(time_range.keys(), time_range.values()):
        st.write(f"Audio unleveled from {start} to {end}")
        data = insert_data(data, duration=start, qc_pnt="Audio unleveled", type="VO", remarks=f"Audio unleveled from {start} to {end}")

    for idx, frame in enumerate(glob("./image_frames/*.jpg")):
        timestamp = frame.split("\\")[-1].split(".jpg")[0].split("frame")[1]

        if idx == 0:
            diff, conf = detect_logo_position(frame)
            st.write(f"Logo detected: confidence score {conf.item()}")
            if abs(diff) < 4:
                st.info("Logo position is standard")
            else:
                st.warning("Logo position is Off (not standard)")
                data = insert_data(data, duration=timestamp, qc_pnt="Logo off", type="Visual", remarks=f"logo off by {diff}")
                # data.loc[len(data)] = [timestamp, "Logo off", f"logo off by {diff}"]

            with st.spinner("Testing Audio Quality..."):
                acc, power = predict_accent(input_video)
            st.info(f"The detected Voice over accent is {acc}, with intensity {power}")
            if acc != "indian":
                data = insert_data(data, duration=timestamp, qc_pnt=f"Accent is {acc}",type="VO", remarks=f"intensity is {power}")
                # data.loc[len(data)] = [timestamp, f"Accent is {acc}", f"intensity is {power}"]
                # data.reset_index(drop=True, inplace=True)
            

        ethnicity = predict_ethnicity_from_image(frame)
        fonttype = get_ost_font_type(frame)

        if len(ethnicity):
            ethnics = [eth["label"] for eth in ethnicity]
            st.write(f"{timestamp}: Detected faces: {ethnics} | Font type: {fonttype}")

            if set(ethnics).intersection(set(["black", "white", "asian", "others"])) != set():
                data = insert_data(data, duration=timestamp, qc_pnt=f"Ethnicity detected: {ethnics}", type="Visual")
                # data.loc[len(data)] = [timestamp, f"Ethnicity detected: {ethnics}", ""]
        else:
            st.write(f"{timestamp}: Font type: {fonttype}")

        if fonttype == "others":
            data = insert_data(data, duration=timestamp, qc_pnt=f"Font type: {fonttype}", type="Visual")
            # data.loc[len(data)] = [timestamp, f"Font type: {fonttype}", ""]
        
    return data