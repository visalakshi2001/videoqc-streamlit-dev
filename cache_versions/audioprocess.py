from glob import glob
import streamlit as st
import io
import os
from datetime import datetime
import pandas as pd
import tempfile

from tests import (predict_accent, get_audio_patchwork,
                    get_audio_stats, detect_noise_and_background)


def audio_processing(batch):

    if "flag" not in st.session_state:
        st.session_state["flag"] = 0
    if "hash" not in st.session_state:
        st.session_state["hash"] = {}
    if "checkpoint" not in st.session_state:
        st.session_state["checkpoint"] = pd.DataFrame()

    if batch == "Single Unit":
        uploaded = st.file_uploader("Upload an Audio file", type=["mp3", "wav"])

        if uploaded is not None:

            with st.columns([2,2])[0]:
                st.audio(uploaded)
            
            # read uploaded file
            st.session_state["vid_name"] = uploaded.name
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())

            expander_object = st.expander(("📢  📂📁 " + uploaded.name + " Is being parsed... ⏬"))
            st.session_state["expander_object"] = expander_object
            data = run_audio_qc_test_single(tfile.name)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                data.to_excel(writer, sheet_name='QCSheet1')

                # Close the Pandas Excel writer and output the Excel file to the buffer
                writer.save()

                st.download_button (
                label="Download Excel QCSheet",
                data=buffer,
                file_name="audioqcreport.xlsx",
                mime="application/vnd.ms-excel"
                )

                st.session_state["stop"] = True
                st.session_state["start"] = False
    
    # --------------------------------------------------------------------------
    elif batch == "Bulk Units":
        
        # receive project path stored from infoform() submission
        folder = st.session_state["project_path"]


        topics = glob(folder + "/**/**.mp3", recursive=True) + glob(folder + "/**/**.wav", recursive=True)
        names = [topic.split("/")[-1].split("\\")[-1] for topic in topics]
        with st.expander("Detected Audios"):
                for name in names:
                    st.text(name)
        
        st.warning(f"Total audio files to process: {len(topics)}")

        stop = st.button("Stop Exec")

        if stop: 
            st.warning("Stopping... Please wait until you see a DOWNLOAD BUTTON before pressing anything", icon="❗")
        # print("stob_btn above me")
        
        flag = st.session_state["flag"]

        data = {}
        df = pd.DataFrame(columns = ["Session No", "Topic Name", "Duration", "Developed by", 
                                    "Reported by", "Date", "QCPoint", "Type", "remarks", "Path"])
        
        st.write(f"QC on {folder}")
        progressbar = st.progress(0)
        i=0  

        for audio in glob(folder + "/**/**.mp3", recursive=True) + glob(folder + "/**/**.wav", recursive=True):
            
            st.success(audio, icon="📌")
            topic_path = audio.split("\\")[-1]
            st.session_state["session_name"] = topic_path

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
                    file_name="audioqcreport.xlsx",
                    mime="application/vnd.ms-excel"
                    )

                break
            # else inner loop continues. no break statement inside inner loop, 
            # as app resets after stop button press

            vid_name = audio.split("\\")[-1].split(".mp3")[0].split(".wav")[0]
            st.session_state["vid_name"] = vid_name

            # st.success("📂📁" + video + " Is being parsed... ⏬", icon="📢")
            expander_object = st.expander(("📢  📂📁 " + audio + " Is being parsed... ⏬"))
            st.session_state["expander_object"] = expander_object
            
            ##########################

            # run qc test on video and feed into dataframe
            df = df.append(run_audio_qc_test_single(audio)).reset_index(drop=True)

            data[f"{flag}"] = audio
            st.session_state["checkpoint"] = df

            flag += 1
            st.session_state["flag"] = flag
            st.session_state["hash"] = data

            progressbar.progress((i + 1)/len(topics))
            i = i+1

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
                file_name="audioqcreport.xlsx",
                mime="application/vnd.ms-excel"
                )
            st.session_state["stop"] = True
            st.session_state["start"] = False

        return stop



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

def run_audio_qc_test_single(input_audio):


    data = pd.DataFrame(columns = ["Session No", "Topic Name", "Duration", "Developed by", "Reported by",
                                    "Date", "QCPoint", "Type", "remarks", "Path"])
    expander_object = st.session_state["expander_object"]

    with st.spinner("Testing Audio Quality..."):
        
        acc, power = predict_accent(input_audio)
        bgm_bgn = detect_noise_and_background(input_audio)
        expander_object.info(f"The detected Voice over accent is {acc}, with intensity {power}")
        expander_object.info(f"The BG Music Status is {bgm_bgn['bgm']} ") # |  BG Noise Status is {bgm_bgn['bgn']}
        if acc != "indian":
            if power > 70:
                data = insert_data(data, duration="0:00:00", qc_pnt=f"Accent is {acc}",type="VO", remarks=f"intensity is {power}")

        if bgm_bgn["bgm"] != "both":
            data = insert_data(data, duration="0:00:00", qc_pnt=f"The BG Music has only {bgm_bgn['bgm']}",type="Audio", remarks=f"can be Audio Level issue")
#         if bgm_bgn["bgn"] != "Clean":
#             data = insert_data(data, duration=timestamp, qc_pnt=f"The BG is Noisy",type="Audio", remarks=f"can be Audio Level issue")

        # Extract Audio status
        time_range = get_audio_patchwork(input_audio)
        
        for start, end in zip(time_range.keys(), time_range.values()):
            expander_object.write(f"Audio unleveled from {start} to {end}")
            data = insert_data(data, duration=start, qc_pnt="Audio unleveled", type="VO", remarks=f"Audio unleveled from {start} to {end}")

        result, complexity_scale, stat_dict = get_audio_stats(input_audio)

        expander_object.info(f" PACE: {stat_dict['Pace (WPM)']}  |  Lexical Complexity: {stat_dict['Lexical Complexity Score']}  |  Apparent Difficult words: {stat_dict['Difficult words']}")
        if stat_dict["Pace (WPM)"] > 185:
            data = insert_data(data, duration="0:00:00", qc_pnt="Pace too fast", type="VO", remarks=f"Pace of VO is {stat_dict['Pace (WPM)']}")
        if stat_dict["Lexical Complexity Score"] < 6 or stat_dict["Lexical Complexity Score"] > 8:
            data = insert_data(data, duration="0:00:00", qc_pnt="Language difficulty exists", type="VO", remarks=f"Lexical Complexity is {stat_dict['Lexical Complexity Score']}")
            data = insert_data(data, duration="0:00:00", qc_pnt=f"Apparent Difficult words: {stat_dict['Difficult words']}", type="VO", remarks=f"In case the words are not subjectively difficult, ignore this error.")
        
            
    
    return data
