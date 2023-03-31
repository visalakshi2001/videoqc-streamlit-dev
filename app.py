import streamlit as st
import pandas as pd
from cache_versions.singleapp import single_processing
from cache_versions.batchprocess import batch_processing
from cache_versions.settings import settings

st.set_page_config(page_title="VideoQC",
                   page_icon="🎬", layout="wide")


def start_app(pages: list, page: str):
    
    st.markdown("<center><h1>🎥 VideoQC Application Demo</h1></center>", unsafe_allow_html=True)

    main, setting_tab = st.tabs(["Main", "Settings"])

    if "start" not in st.session_state:
        st.session_state["start"] = False
    

    with main:
        if page == pages[1]:
            infobtn = info_form(1)
            if infobtn:
                set_process_initiation(True)
            if st.session_state["start"]:
                single_processing()
        elif page == pages[2]:
            infobtn = info_form(2)
            if infobtn:
                set_process_initiation(True)
            if st.session_state["start"]:  
                batch_processing()
    
    with setting_tab:
        settings()

def sidebar(pages):


    pageidx = st.session_state["page"]

    with st.sidebar:
        
        st.title("Demo VideoQC for:")
        
        page = st.radio("Select analysis", 
                        options = pages)

        st.session_state["page"] = pages.index(page)
        pageidx = st.session_state["page"]
        page = pages[pageidx]

    return page

        

def info_form(state=1):

    if "qc_mem_name" not in st.session_state:
        st.session_state["qc_mem_name"] = ""
    if "qc_dev_mem" not in st.session_state:
        st.session_state["qc_dev_mem"] = ""
    if "project_path" not in st.session_state:
        st.session_state["project_path"] = ""
    if "session_name" not in st.session_state:
        st.session_state["session_name"] = ""


    with st.form("info_input_form"):
        cols = st.columns(2)
        with cols[0]:
            qc_mem_name = st.text_input("QC Member Name", placeholder="Write down the QC Member name")
            project_path = st.text_input("Path to project folder", placeholder=r"network_url/Projects/[Project_Name]/")
        with cols[1]:
            qc_dev_mem = st.text_input("Video Developed by", placeholder="Write down the QC Video Developer name")
            if state == 1:
                session_name = st.text_input("Session Topic Name", placeholder=r"Write the Topic you want to process inside ./02_Animation/")
        infobtn = st.form_submit_button("Continue")

        if infobtn:
            st.session_state["qc_mem_name"] = qc_mem_name
            st.session_state["qc_dev_mem"] = qc_dev_mem
            st.session_state["project_path"] = project_path
            if state==1:
                st.session_state["session_name"] = session_name

        return infobtn

def set_process_initiation(val):
    st.session_state["start"] = val

if __name__ == "__main__":

    pages = ["ScriptQC", "Single Video processing", "Batch processing", "AudioQC"]

    if "page" not in st.session_state:
        st.session_state["page"] = 1

    page = sidebar(pages)
    start_app(pages, page)
