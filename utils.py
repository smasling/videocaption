import json
from os import listdir
import cv2
import numpy as np


def conv_video_to_frames(s, train, count):
    cap = cv2.VideoCapture(s)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frameCount, frameHeight, frameWidth )
    buf = np.empty((int(frameCount/8), frameHeight, frameWidth, 3), np.dtype('uint8'))
    fff = 0
    ret = True
    while (fff < frameCount and ret):
        ret, temp = cap.read()
        if fff % 8 == 0:
            buf[fff] = temp
        fff += 1
    cap.release()
    curr = np.pad(buf, (0, (113 - buf.shape[0]), (0,0), (0,0), (0,0)))
    del buf
    del curr
    print(count)
    train[count] += curr



def create_train_videos():
    vids = [f[:-4] for f in listdir('testFolder')]
    train = np.zeros((len(vids), 113, 240, 320, 3), dtype=np.uint8)
    count = 0
    for v in vids:
        file = "testFolder/" + v + ".mp4"
        ret[v] = conv_video_to_frames(file, train, count)
        count += 1

    np.save('train', train)






create_train_videos()


def load_video_annotations(s):
    with open(s) as w:
        obj = json.load(w)
        videos = obj['videos']
        sentences = obj['sentences']
    return (videos, sentences)

def create_mini_split():
    vids = [f[:-4] for f in listdir('testFolder')]
    videos, sentences = load_video_annotations('train_val_videodatainfo.json')
    mini_train = {}
    for v in vids:
        mini_train[v] = {'sentences': []}
        for vid in videos:
            if vid['video_id'] == v:
                mini_train[v]['vid_info'] = vid
        for sentence in sentences:
            if sentence['video_id'] == v:
                mini_train[v]['sentences'].append(sentence)
    return mini_train

# videodata = skvideo.io.vread("testFolder/video12.mp4")
# print(videodata.shape)