# ETHNICITY DETECTION
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Activation, BatchNormalization, Add
from keras import Model
from keras.models import Sequential


# ETHNICITY DETECTION
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


    return model

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
    model = getEthModel()
    model.load_weights("models/4_29.h5")

    image = cv2.imread(img_inp)

    if image.shape == (200,200,3):
        y_hat = model.predict(np.asarray([image]))
        label = rev_labels(np.argmax(y_hat))
        result_list = [{'box': [0, 0, 200, 200], 'label': label}]
        # draw_image_with_boxes(img_path, result_list)
        return result_list
    else:
        detector = MTCNN()
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
    logo_detector = YOLO("models/best.pt")

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

# FONT DETECTION
from keras.models import load_model
import tensorflow as tf

# FONT DETECTION
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

    return model

# FONT DETECTION
def get_font_type(img_inp):
    # model = load_model("models/fontclassifier.h5", compile=False)
    model = getFontModel()
    model.load_weights("models/fontclassifierweights.hdf5")
    
    image = cv2.imread(img_inp)

    image = tf.image.resize(image, [256, 256])
    yhat = model.predict(np.expand_dims(image/255, 0))

    print(img_inp, yhat)

    pred = ""
    if yhat >= 0.5:
        pred = "segoeui"
    elif yhat >= 0.3 and yhat < 0.5:
        pred = "others"
    else:
        pred = "no font"
    return pred

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
LANG_DICT = {'india': 0, 'uk': 1, 'usa': 2}

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

