import os
import cv2
import numpy as np
from datetime import timedelta
from pathlib import Path
import streamlit as st

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 20 / saving_fps):
        s.append(i)
    return s

@st.cache(suppress_st_warning=True)
def extract_frames_main(video_file, storage_path:str = "./", SAVING_FRAMES_PER_SECOND = 15):

    storage_path = os.path.join(storage_path, "image_frames/")
    if not os.path.exists(storage_path):
        os.mkdir(storage_path) 
    
    if len(os.listdir(storage_path)) > 0:
        [f.unlink() for f in Path(storage_path).glob("*") if f.is_file()]
    
    # read the video file    
    test_video = cv2.VideoCapture(video_file)
    # get the FPS of the video
    frameRate = test_video.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    if st.session_state["frame_rate"] != 15:
        saving_frames_per_second = min(frameRate, SAVING_FRAMES_PER_SECOND)
    else:
        saving_frames_per_second = 15
    
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap=test_video, saving_fps=saving_frames_per_second)

    # Count for our frames
    frameCnt = 0
    
    print(f"Total frames to be extracted: {len(saving_frames_durations)}")
    while True:
        is_read, frame = test_video.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = frameCnt / frameRate
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            name = storage_path + 'frame'+ frame_duration_formatted +'.jpg'

            cv2.imwrite(name, frame) 

            # print progress to console
            if int(frame_duration) % 50 == 0:
                print("Extracting frames .." + str(int(frame_duration)))
            
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        frameCnt += 1

    test_video.release()
    cv2.destroyAllWindows()

    return