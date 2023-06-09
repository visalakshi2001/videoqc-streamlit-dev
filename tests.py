import streamlit as st

# ETHNICITY DETECTION
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Activation, BatchNormalization, Add
from keras import Model
from keras.models import Sequential


# ETHNICITY DETECTION
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def getEthModel(vesion=4):
    
    inp = Input(shape=(200, 200, 3,))

    net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(inp)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=256, kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=256, kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Flatten()(net)

    net = Dense(256, activation='relu')(net)
    net = Dense(512, activation='relu')(net)
    out = Dense(5, activation='softmax')(net)

    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("models/ethnicity_model_4_29.h5")
    detector = MTCNN()

    return model, detector

def rev_labels(pred):
    if pred == 0 :
        return 'white'
    elif pred == 1:
        return 'black'
    elif pred == 2:
        return 'asian'
    elif pred == 3:
        return 'indian'
    elif pred == 4:
        return 'others'

def predict_ethnicity_from_image(img_inp):
    model, detector = getEthModel()

    image = cv2.imread(img_inp)

    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face["box"]
        new_img = cv2.resize(image[y:y+h, x:x+w], (200,200))
        y_hat = model.predict(np.asarray([new_img]))
        label = rev_labels(np.argmax(y_hat))
        face["label"] = label
    # draw_image_with_boxes(img_path, faces)
    return faces

# LOGO DETECTION
from ultralytics import YOLO

# LOGO DETECTION
def detect_logo_position(img_inp):
    logo_detector = YOLO("models/logo_model.pt")

    RES = (720, 1280)
    BOX = (120, 180)
    x, y, w, h = RES[0], RES[1], BOX[0], BOX[1]

    POS = np.array([[121,  60,  54,  56]], dtype='float')

    image = cv2.imread(img_inp)
    if image.shape != (120, 180, 3):
        cropped = image[x-w:x, y-h:y]
    else:
        cropped = image
    
    result = logo_detector(cropped)
    confidence = result[0].boxes.conf

    diff = result[0].boxes.xywh - POS

    return sum(diff[0]).item(), confidence

# FONT AND OST DETECTION
from keras.models import load_model
import tensorflow as tf
import pytesseract
import spacy
from spacy.matcher import Matcher
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except:
    nltk.download("averaged_perceptron_tagger")


# TEXT CORRECTION AND ERROR DETECTION
@st.cache(suppress_st_warning=True)
def detect_text_error(ost):

    status = {}

    print("Checking voice of the sentence")
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    passive_rule = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
    matcher.add('Passive', [passive_rule])

    doc = nlp(ost)
    matches = matcher(doc)
    if len(matches) > 0:
        status["voice"] = "Passive"
    else:
        status["voice"] = "Active"
    
    print("Checking spelling errors")
    blob = TextBlob(ost.lower())
    misspelled = set()
    for word in blob.words:
        if not word.spellcheck()[0][1] == 1.0:
            misspelled.add(str(word))
    if len(misspelled) > 0:
        status["spelling"] = misspelled
    else:
        status["spelling"] = "Good"

    print("Checking uppercase issues with prep and conj")
    stops = set(stopwords.words("english"))
    blob = TextBlob(ost)
    status["hasuppersubord"] = False
    for wordtag in blob.tags:
        if not wordtag[1].startswith("N") and wordtag[0][0].isupper():
            status["hasuppersubord"] = True
    
    print("Checking &")
    if "&" in ost:
        status["contains&"] = True
    else:
        status["contains&"] = False
    
    print("Checking for Orphan words")
    if ost.split("\n")[-1] == ost.split("\n")[-1].split()[-1]:
        status["orphan"] = True
    else:
        status["orphan"] = False
    
    print("Checking Sentence formation")

    return status



# FONT DETECTION
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def getFontModel():
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.load_weights("models/fontclassifierweights.hdf5")

    return model


# FONT AND OST DETECTION
def get_ost_font_type(img_inp):

    # font_model = load_model("models/fontclassifier.h5", compile=False)
    font_model = getFontModel()
    
    ost_model = YOLO("models/ost_model.pt")

    image = cv2.imread(img_inp)
    results = ost_model(image)
    if len(results[0].boxes) > 0:
        print("OST Detected")
        boxes = results[0].boxes.xyxy
        ocr_text = ""
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            try:
                pad = 50
                ocr_text = ocr_text + " " + pytesseract.image_to_string(image[y1:y2, x1-pad:x2+pad])

            except:
                pad = 0
                ocr_text = ocr_text + " " + pytesseract.image_to_string(image[y1:y2, x1-pad:x2+pad])
        
        ocr_text = ocr_text.strip()
        ocr_text = ocr_text.strip("\n")
        ocr_text = ocr_text.rstrip("}{)(-").strip()
        
        if ocr_text != "":
            status = detect_text_error(ocr_text)
        
            image = tf.image.resize(image, [256, 256])
            yhat = font_model.predict(np.expand_dims(image/255, 0))

            print(img_inp, yhat)

            fonttype = ""
            if yhat >= 0.5:
                fonttype = "segoeui"
            elif yhat >= 0.3 and yhat < 0.5:
                fonttype = "others"
            else:
                return (ocr_text, "None, This is only BG Text", status)
            
            return (ocr_text, fonttype, status)
        else:
            return "no OST"

    return "no OST"

# ACCENT CLASSIFICATION
from collections import Counter
import librosa
from pydub import AudioSegment
import os

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10 #35#250
LANG_DICT = {'indian': 0, 'uk': 1, 'usa': 2}

# ACCENT CLASSIFICATION
def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def segment_one(mfcc):
    '''
    Creates segments from one mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def rev_conv_labels(pred, label_dict):
    return list(label_dict.keys())[list(label_dict.values()).index(pred)]

# ACCENT CLASSIFICATION
def predict_accent(video_inp):

    # if video_inp.endswith(".mp4"):
    #     mp4_ext = AudioSegment.from_file(video_inp, "mp4")
    #     extracted_audio = os.path.join(os.getcwd(), "audioextract.mp3")
    #     mp4_ext.export(extracted_audio, "mp3")

    series, sr = librosa.load(video_inp)
    series = librosa.core.resample(y=series,orig_sr=sr,target_sr=RATE, scale=True)
    mfcc = to_mfcc(series)
    segment = segment_one(mfcc)

    model = load_model("models/accent_model_valacc40.h5")

    y_hat = model.predict(segment)
    
    pred = Counter(np.argmax(y_hat, axis=1)).most_common(1)[0][0]
    intensity = Counter(np.argmax(y_hat, axis=1)).most_common(1)[0][1]

    accent = rev_conv_labels(pred, LANG_DICT)
    
    return(accent, intensity/len(y_hat))



# DETECT BGMUSIC AND BGNOISE
def extract_features(audio_data):
    series, sr = librosa.load(audio_data)
    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=series, sr=sr, n_mfcc=40)
    # Calculate the mean and standard deviation of each MFCC coefficient
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    return np.concatenate((mfccs_mean, mfccs_std), axis=0)

def load_audio_models():
    noise_model = load_model("models/noise.h5", compile=False)
    bgm_model = load_model("models/bgm_classifier.h5", compile=False)

    return noise_model, bgm_model

# DETECT BGMUSIC AND BGNOISE
def detect_noise_and_background(audio_file):
    features = extract_features(audio_file)
    
    noise_model, bgm_model = load_audio_models()

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    # Use the trained model to make a prediction
    has_noise = noise_model.predict(features)
    has_bgm = bgm_model.predict(features)

    BGM_LABELS = {
    0: "bgm",
    1: "both",
    2: "vocal",
    }

    BGN_LABELS = {
    0: "Clean",
    1: "Noise",
    }

    return { "bgm": BGM_LABELS[np.argmax(has_bgm)], 
             "bgn": BGN_LABELS[np.argmax(has_noise)]
            }

# PATCHWORK, LEVELS, BG MUSIC LEVEL
import opensmile
import datetime

# PATCHWORK, LEVELS, BG MUSIC LEVEL
def get_cross_corr(audio_file, segment_len = 2):
    # Load the audio file
    audio, sr = librosa.load(audio_file)

    # Define segment length in seconds
    segment_length = segment_len

    # Divide the audio into segments
    n_segments = len(audio) // (segment_length * sr)
    segments = np.array_split(audio, n_segments)

    # Extract OpenSMILE features for each segment
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=1,
        )
    feature_values = []
    for segment in segments:
        features = smile.process_signal(segment, sr)
        feature_values.append(features)
    feature_values = np.array(feature_values)

    # Calculate cross-correlation between adjacent segments by taking the max of feature map correlation
    max_corr_values = np.zeros(n_segments-1)
    for i in range(n_segments-1):
        corr = np.abs(np.correlate(feature_values[i][0], feature_values[i+1][0], mode='same'))
        max_corr_values[i] = np.max(corr)
    
    # plot cross-correlation 
    timestamps = np.arange(n_segments-1) * segment_length


    return max_corr_values, timestamps

def calculate_differences(corr_vals, end_second = 10):
    differences = {}
    
    # Calculate differences between consecutive positions
    for i in range(len(corr_vals) - 1):
        diff = []
        for j in range(1, end_second):
            try:
                diff.append(round(corr_vals[i + j] - corr_vals[i], 2))
            except:
                continue
        differences[str(corr_vals[i])] = diff
    
    return differences

def catch_timestamps(differences, corr_values, timestamps, threshold=0.18):
    time_catch = {}
    for i, corr_val in enumerate(corr_values):
        if i == len(corr_values) -1 :
            continue
        # convert current timestamp to mm:ss format
        t = str(datetime.timedelta(seconds=int(timestamps[i])))[-5:]

        # capture differences that are >0.10 and note their timestamp in relative seconds difference (index value)
        flags = [differences[str(corr_val)].index(diff) for diff in differences[str(corr_val)] if diff>=threshold]

        time_catch[timestamps[i]] = flags
    
    time_range = {}
    for start, ends in zip(time_catch.keys(), time_catch.values()):

        if ends != []:
            s = str(datetime.timedelta(seconds=int(start)))[-5:]
            end = max(ends)
            e = str(datetime.timedelta(seconds=int(start + end)))[-5:]
            # print(s, "--", e)
            time_range[s] = e
    return time_range

def get_audio_patchwork(video_inp):

    corr_values, timestamp = get_cross_corr(video_inp, segment_len=2)

    diffs = calculate_differences(corr_vals=corr_values/1e7)

    time_range = catch_timestamps(differences=diffs, corr_values=corr_values/1e7, timestamps=timestamp)

    return time_range

# EXTRACT TRANSCRIPT
import whisper
import textstat
import pandas as pd

# EXTRACT TRANSCRIPT
@st.cache(suppress_st_warning=True)
def get_transcript_model(model=None):
    if model is None:
        whisper_model = whisper.load_model("tiny")
    
    return whisper_model

def get_audio_stats(audio_file, model = None, calculate=True):
    
    save_transcript = st.session_state["save_transcript"]
    
    whisper_model = get_transcript_model()
    print("Whisper model loaded..")

    result = whisper_model.transcribe(audio_file)
    if save_transcript:
        print("Transcripting and Saving script..")
        with open(f"{st.session_state['vid_name']}_transcript.txt", "w+") as f:
            f.write("Video: " + st.session_state["vid_name"])
            f.write(result["text"])
    
    total_activity_time = 0
    voice_activity_terms = []
    complexity_scale = pd.DataFrame(columns=["start", "end", "text", "score"])
    for segment in result["segments"]:
        voice_activity_terms.extend(segment["text"].split())
        total_activity_time += (segment["end"] - segment["start"])
        complexity_scale.loc[len(complexity_scale)] = [segment["start"], segment["end"], segment["text"], textstat.dale_chall_readability_score(segment["text"])]

    stat_dict = {
        "Pace (WPM)": (len(voice_activity_terms)/total_activity_time)*60,
        "Lexical Complexity Score": complexity_scale["score"].mean(),
        "Complexity Level": textstat.text_standard(result["text"]),
        "Difficult words": [textstat.difficult_words_list(result["text"])],
    }

    return result, complexity_scale, stat_dict