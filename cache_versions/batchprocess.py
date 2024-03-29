from glob import glob
import streamlit as st
import io
import os
from datetime import datetime
import time
import pandas as pd
# from stqdm import stqdm
from tqdm import tqdm
import tempfile
from videoparser import extract_frames_main
from tests import detect_logo_position, predict_ethnicity_from_image, get_ost_font_type, predict_accent
from cache_versions.singleapp import run_video_qc_test_single

def batch_processing():

    # # to view the progress of app
    # st.write(st.session_state)

    st.markdown("<center><h3>Batch Video Processing</h3></center>", unsafe_allow_html=True)
    
    topics = []

    # receive project path stored from infoform() submission
    folder = st.session_state["project_path"]

    # target the main video folder
    folder = os.path.join(folder, "02_Animation")

    # Test whether every video is parsed in order
    with st.expander("Detected Topics"):
        for topic in os.listdir(folder):
            st.text(topic)

    with st.expander("Detected Videos"):
        for topic in os.listdir(folder):
            name = topic
            topic = os.path.join(folder, topic)
            topics.append(topic)
            st.text(f"{name} : {os.listdir(topic)}")
            

    st.warning(f"Total topics to process: {len(topics)}")
    # total_t = st.empty()

    stop = st.button("Stop Exec")

    if stop: 
        st.warning("Stopping... Please wait until you see a DOWNLOAD BUTTON before pressing anything", icon="❗")
    # print("stob_btn above me")
    if "flag" not in st.session_state:
        st.session_state["flag"] = 0
    if "hash" not in st.session_state:
        st.session_state["hash"] = {}
    if "checkpoint" not in st.session_state:
        st.session_state["checkpoint"] = pd.DataFrame()
    
    flag = st.session_state["flag"]

    data = {}
    df = pd.DataFrame(columns = ["Session No", "Topic Name", "Duration", "Developed by", 
                                 "Reported by", "Date", "QCPoint", "Type", "remarks", "Path"])

    st.write(f"QC on {folder}")
    progressbar = st.progress(0)
    i=0
    for topic in topics:
        st.success(topic, icon="📌")
        topic_path = topic.split("\\")[-1]
        st.session_state["session_name"] = topic_path
        # time.sleep(2)

        if stop:
            df = st.session_state["checkpoint"]
            print("Stopping")
            st.session_state["stop"] = stop
            st.session_state["start"] = False

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                df.to_excel(writer, sheet_name='QCSheet1')

                # Close the Pandas Excel writer and output the Excel file to the buffer
                writer.save()

                st.download_button(
                label="Download Excel QCSheet",
                data=buffer,
                file_name="qcreport.xlsx",
                mime="application/vnd.ms-excel"
                )

            break
        # else inner loop continues. no break statement inside inner loop, 
        # as app resets after stop button press
        for video in glob(topic + "/*.mp4"):
            vid_name = video.split("\\")[-1].split(".mp4")[0]
            st.session_state["vid_name"] = vid_name

            # st.success("📂📁" + video + " Is being parsed... ⏬", icon="📢")
            expander_object = st.expander(("📢  📂📁 " + video + " Is being parsed... ⏬"))
            st.session_state["expander_object"] = expander_object
            
            ##########################

            # run qc test on video and feed into dataframe
            df = df.append(run_video_qc_test_single(video)).reset_index(drop=True)

            data[f"{flag}"] = video
            st.session_state["checkpoint"] = df

            flag += 1
            st.session_state["flag"] = flag
            st.session_state["hash"] = data
            progressbar.progress((i + 1)/len(glob(topic + "/*.mp4") * len(topics)))
            i = i+1
            # time.sleep(5)
    
    # after executing all videos, save qc sheet
    if not stop: 
        df = st.session_state["checkpoint"]
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            df.to_excel(writer, sheet_name='QCSheet1')

            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()

            st.download_button(
            label="Download Excel QCSheet",
            data=buffer,
            file_name="qcreport.xlsx",
            mime="application/vnd.ms-excel"
            )
            st.session_state["stop"] = True
            st.session_state["start"] = False

    return stop

