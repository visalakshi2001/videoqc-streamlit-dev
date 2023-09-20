from glob import glob
import streamlit as st
import io
import os
from datetime import datetime
import pandas as pd
import docx

from tests import check_scripts


def script_processing(batch):

    if "flag" not in st.session_state:
        st.session_state["flag"] = 0
    if "hash" not in st.session_state:
        st.session_state["hash"] = {}
    if "checkpoint" not in st.session_state:
        st.session_state["checkpoint"] = pd.DataFrame()

    if batch == "Single Unit":
        uploaded = st.file_uploader("Upload a Sciprt file", type=["doc", "docx"])

        if uploaded is not None:
            exp = st.expander(("📢  📂📁 " + uploaded.name + " Content... ⏬"))
            with st.columns([2,2])[0]:
                content = docx.Document(uploaded)
                for para in content.paragraphs:
                    if para.text != "":
                        exp.write(para.text.strip())
            
            
            # read uploaded file
            st.session_state["vid_name"] = uploaded.name

            expander_object = st.expander(("📢  📂📁 " + uploaded.name + " Is being parsed... ⏬"))
            st.session_state["expander_object"] = expander_object
            data = run_script_qc_test_single(uploaded)

            if len(data) > 0:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Write each dataframe to a different worksheet.
                    data.to_excel(writer, sheet_name='QCSheet1')

                    # Close the Pandas Excel writer and output the Excel file to the buffer
                    writer.save()

                    st.download_button (
                    label="Download Excel QCSheet",
                    data=buffer,
                    file_name="scriptqcreport.xlsx",
                    mime="application/vnd.ms-excel"
                    )
            else:
                st.warning("No report generated. Script is clean")
            
            st.session_state["stop"] = True
            st.session_state["start"] = False
    
#     # --------------------------------------------------------------------------
    elif batch == "Bulk Units":


        # receive project path stored from infoform() submission
        folder = st.session_state["project_path"]


        topics = glob(folder + "/**/**.docx", recursive=True) + glob(folder + "/**/**.doc", recursive=True)
        names = [topic.split("/")[-1].split("\\")[-1] for topic in topics]
        with st.expander("Detected Scripts"):
                for name in names:
                    st.text(name)
        
        st.warning(f"Total script files to process: {len(topics)}")

        stop = st.button("Stop Exec")

        if stop: 
            st.warning("Stopping... Please wait until you see a DOWNLOAD BUTTON before pressing anything", icon="❗")
        # print("stob_btn above me")
        
        flag = st.session_state["flag"]

        data = {}
        df = pd.DataFrame(columns = ["Session No", "Topic Name", "Sentence", "Developed by", 
                                    "Reported by", "Date", "QCPoint", "Type", "remarks", "Path"])
        
        st.write(f"QC on {folder}")
        progressbar = st.progress(0)
        i=0  

        for script in glob(folder + "/**/**.docx", recursive=True) + glob(folder + "/**/**.doc", recursive=True):
            
            st.success(script, icon="📌")
            topic_path = script.split("\\")[-1]
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
                    file_name="scriptqcreport.xlsx",
                    mime="application/vnd.ms-excel"
                    )

                break
            # else inner loop continues. no break statement inside inner loop, 
            # as app resets after stop button press

            vid_name = script.split("\\")[-1].split(".mp3")[0].split(".wav")[0]
            st.session_state["vid_name"] = vid_name

            # st.success("📂📁" + video + " Is being parsed... ⏬", icon="📢")
            expander_object = st.expander(("📢  📂📁 " + script + " Is being parsed... ⏬"))
            st.session_state["expander_object"] = expander_object
            
            ##########################

            # run qc test on video and feed into dataframe
            df = df.append(run_script_qc_test_single(script)).reset_index(drop=True)

            data[f"{flag}"] = script
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
                file_name="scriptqcreport.xlsx",
                mime="application/vnd.ms-excel"
                )
            st.session_state["stop"] = True
            st.session_state["start"] = False

        return stop



def insert_data(dataframe,  sentence,  qc_pnt, type, remarks=""):

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
    dataframe.loc[len(dataframe)] = [project_name, topic_name, sentence, qc_dev, qc_mem,
                                     datetime.today().strftime("%d-%m-%Y"), qc_pnt, type, remarks, path]
    
    return dataframe

def run_script_qc_test_single(input_script):


    data = pd.DataFrame(columns = ["Session No", "Topic Name", "Sentence", "Developed by", "Reported by",
                                    "Date", "QCPoint", "Type", "remarks", "Path"])
    expander_object = st.session_state["expander_object"]

    with st.spinner("Testing Script Quality..."):
        results = check_scripts(input_script)

    words = results["words"]
    sents = results["sents"]
    expander_object.info(f"Word Count: {words}   |   Sentence Count: {sents}")
    data = insert_data(data, sentence="", qc_pnt=f"Word Count: {words}", type="Script", remarks=f"")
    data = insert_data(data, sentence="", qc_pnt=f"Sentence Count: {sents}", type="Script", remarks=f"")

    plag = results["plag"]

    if plag != []:
        for detection in plag:
            # deleting highlights key from the matches
            [match.pop("highlight", None) for match in detection["matches"]]

            expander_object.write(f"PLAGIARISM: \ntext: {detection['text']} \nscore: {detection['plagiarism']} \nmatches: {detection['matches']}")

            data = insert_data(data, sentence=detection["text"], 
                               qc_pnt=f"Plagiarism Detected {detection['plagiarism']}", 
                               type="Script", remarks=f"Matches from {detection['matches']}")
    else:
        expander_object.write("No plagiarism found")
    
    gram = results["gram"]

    if gram != []:
        for err in gram:
            if "err" in err:
                expander_object.write(f"GRAMMAR / SPELLING ERROR: \ntext: {err['err']} \
                                    \correction: {err['corr']}")
                data = insert_data(data, sentence=err['err'], qc_pnt=f"grammar or spelling error",type="Script", remarks=f"Possible correction: {err['corr']}")
            if "is_pass" in err:
                expander_object.write(f"Passive Sentence: \ntext: {err['is_pass']}")
                data = insert_data(data, sentence = err["is_pass"], qc_pnt=f"Is Passive",type="Script", remarks="")
            if "sent_length" in err:
                expander_object.write(f"Sentence length too long for \ntext: {err['sent_length']}")
                data = insert_data(data, sentence = err["sent_length"], qc_pnt=f"Is Passive",type="Script", remarks="")
    else:
        expander_object.write("No grammar or spelling problems found")
    
    return data
