import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from dataloader import get_loader
from process_data import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import cv2
import os


def get_buf(movie):
    cap = cv2.VideoCapture(movie)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount/5) + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fff = 0
    ret = True
    while (fff < frameCount and ret):
      ret, temp = cap.read()
      if fff % 5 == 0:
        buf[int(fff/5)] = temp
      fff += 1
    cap.release()


def extract_features(buf):
    resnet = models.video.r3d_18(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:-2])
    return model(buf)

def main():
    ret = {}
    for filename in tqdm(os.listdir('testFolder')):
        print(filename[:-4])
        path = os.path.join('testFolder', filename)
        buf = get_buf(path)
        features = extract_features(buf).data.numpy()
        np.save(features, filename[:-4] + ".npy")









if __name__ == "__main__":
    main()