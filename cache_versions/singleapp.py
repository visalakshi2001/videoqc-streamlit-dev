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
                    get_audio_stats, detect_noise_and_background)



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
        
        expander_object = st.expander(("📢  📂📁 " + uploaded.name + " Is being parsed... ⏬"))
        st.session_state["expander_object"] = expander_object
        data = run_video_qc_test_single(tfile.name)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            data.to_excel(writer, sheet_name='QCSheet1')

            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

            st.download_button (
            label="Download Excel QCSheet",
            data=buffer,
            file_name="qcreport.xlsx",
            mime="application/vnd.ms-excel"
            )

            st.session_state["stop"] = True
            st.session_state["start"] = False
        


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

    
    # extract 
    with st.spinner("🧩 Parsing video frames..."):
        extract_frames_main(input_video)
    
    st.write("Parsing complete, running test cases:")

    data = pd.DataFrame(columns = ["Session No", "Topic Name", "Duration", "Developed by", "Reported by",
                                    "Date", "QCPoint", "Type", "remarks", "Path"])
    expander_object = st.session_state["expander_object"]

    for idx, frame in enumerate(glob("./image_frames/*.jpg")):
        timestamp = frame.split("\\")[-1].split(".jpg")[0].split("frame")[1]
        timestamp = timestamp.replace(".", ":").replace("-", ":")[2:]
        if idx == 0:
            try:
                diff, conf = detect_logo_position(frame)
                st.write(f"Logo detected: confidence score {conf.item()}")
                if abs(diff) < 4:
                    st.info("Logo position is standard")
                else:
                    st.warning("Logo position is Off (not standard)")
                    data = insert_data(data, duration=timestamp, qc_pnt="Logo off", type="Visual", remarks=f"logo off by {diff}")
            except:
                try:
                    frame = glob("./image_frames/*.jpg")[idx+2]
                    diff, conf = detect_logo_position(frame)
                    st.write(f"Logo detected: confidence score {conf.item()}")
                    if abs(diff) < 4:
                        st.info("Logo position is standard")
                    else:
                        st.warning("Logo position is Off (not standard)")
                        data = insert_data(data, duration=timestamp, qc_pnt="Logo off", type="Visual", remarks=f"logo off by {diff}")
                except:
                    st.warning("Logo missing!")
                    data = insert_data(data, duration=timestamp, qc_pnt="Logo missing", type="Visual", remarks=f"logo missing")



        ethnicity = predict_ethnicity_from_image(frame)
        ost_status = get_ost_font_type(frame)

        if ost_status != "no OST" or len(ethnicity):
            # if ost
            if ost_status != "no OST":
                status = ost_status[2]
                expander_object.write(f"{timestamp}: Font type: {ost_status[1]}  |  OST: {ost_status[0]}")

                ost_error_text = ""
                if status["voice"] != "Active":
                    ost_error_text = ost_error_text + " " + "Sentence Voice is Passive"
                if status["spelling"] != "Good":
                    ost_error_text = ost_error_text + " " + f"mispelled word {status['spelling']}"
                if status["hasuppersubord"] != False:
                    ost_error_text = ost_error_text + " " + 'Subordinate has Uppercase'
                if status["contains&"] != False:
                    ost_error_text = ost_error_text + " " + 'Contains &'
                if status["orphan"] != False:
                    ost_error_text = ost_error_text + " " + 'Contains Ophan word'
                
                if ost_error_text !="":
                    data = insert_data(data, duration=timestamp, qc_pnt=ost_error_text, type="OST")
                    
                if ost_status[1] == "others":
                    data = insert_data(data, duration=timestamp, qc_pnt=f"Font type: {ost_status[1]}", type="Visual")
                # data.loc[len(data)] = [timestamp, f"Font type: {fonttype}", ""]

            # if ethnicity
            if len(ethnicity):
                ethnics = [eth["label"] for eth in ethnicity]
                expander_object.write(f"{timestamp}: Detected faces: {ethnics}")
            
                if set(ethnics).intersection(set(["black", "white", "asian", "others"])) != set():
                    data = insert_data(data, duration=timestamp, qc_pnt=f"Ethnicity detected: {ethnics}", type="Visual")
                    # data.loc[len(data)] = [timestamp, f"Ethnicity detected: {ethnics}", ""]

        else:
            expander_object.write(f"{timestamp}: No QC Point")
    
    if st.session_state["do_audioqc"]:
        # Extract Audio status
        time_range = get_audio_patchwork(input_video)
        
        for start, end in zip(time_range.keys(), time_range.values()):
            expander_object.write(f"Audio unleveled from {start} to {end}")
            data = insert_data(data, duration=start, qc_pnt="Audio unleveled", type="VO", remarks=f"Audio unleveled from {start} to {end}")

        result, complexity_scale, stat_dict = get_audio_stats(input_video)

        expander_object.info(f" PACE: {stat_dict['Pace (WPM)']}  |  Lexical Complexity: {stat_dict['Lexical Complexity Score']}  |  Apparent Difficult words: {stat_dict['Difficult words']}")
        if stat_dict["Pace (WPM)"] > 185:
            data = insert_data(data, duration="0:00:00", qc_pnt="Pace too fast", type="VO", remarks=f"Pace of VO is {stat_dict['Pace (WPM)']}")
        if stat_dict["Lexical Complexity Score"] < 6 or stat_dict["Lexical Complexity Score"] > 8:
            data = insert_data(data, duration="0:00:00", qc_pnt="Language difficulty exists", type="VO", remarks=f"Lexical Complexity is is {stat_dict['Lexical Complexity Score']}")
            data = insert_data(data, duration="0:00:00", qc_pnt=f"Apparent Difficult words: {stat_dict['Difficult words']}", type="VO", remarks=f"In case the words are not subjectively difficult, ignore this error.")
        
        with st.spinner("Testing Audio Quality..."):
                acc, power = predict_accent(input_video)
                bgm_bgn = detect_noise_and_background(input_video)
                expander_object.info(f"The detected Voice over accent is {acc}, with intensity {power}")
                st.info(f"The BG Music Status is {bgm_bgn['bgm']} ") # |  BG Noise Status is {bgm_bgn['bgn']}
                if acc != "indian":
                    if power > 70:
                        data = insert_data(data, duration=timestamp, qc_pnt=f"Accent is {acc}",type="VO", remarks=f"intensity is {power}")
                if bgm_bgn["bgm"] != "both":
                    data = insert_data(data, duration=timestamp, qc_pnt=f"The BG Music has only {bgm_bgn['bgm']}",type="Audio", remarks=f"can be Audio Level issue")
        #         if bgm_bgn["bgn"] != "Clean":
        #             data = insert_data(data, duration=timestamp, qc_pnt=f"The BG is Noisy",type="Audio", remarks=f"can be Audio Level issue")
            
    
    return data