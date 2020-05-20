import utils
import json
import numpy as np
import os
import nltk
import torch
import torch.utils.data as data
from collections import defaultdict
import torchvision.transforms as transforms
import cv2

class DataLoader(data.Dataset):
  def __init__(self, ids, vocab, transform=None):
    self.ids = ids
    self.vocab = vocab
    self.transform = transform

  def __getitem__(self, index):
    vocab = self.vocab
    caption = self.ids[index][0]
    movie = self.ids[index][1]
    fle = "testFolder/" + movie + ".mp4"


    # image = Image.open(os.path.join(self.root, path)).convert('RGB')
    # if self.transform is not None:
    #     image = self.transform(image)


    cap = cv2.VideoCapture(fle)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('frameWidth', frameWidth)
    print(frameCount, frameHeight, frameWidth )
    buf = np.empty((int(frameCount/10) + 1, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fff = 0
    ret = True
    while (fff < frameCount and ret):
      ret, temp = cap.read()
      if fff % 10 == 0:
        buf[int(fff/10)] = temp
      fff += 1
    cap.release()

    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    return torch.Tensor(buf), target

  def __len__(self):
    return len(self.ids)


def collate_fn(data):
  images, captions = zip(*data)
  _, W, H, C = images[0].shape
  mx = 0
  for i in range(len(images)):
    f = images[i].shape[0]
    if f > mx:
      mx = f
  newImages = torch.zeros(len(images), mx, W, H, C)
  for i, image in enumerate(images):
    newImages[i, :images[i].shape[0]] = image
  images = newImages
  lengths = [len(cap) for cap in captions]
  targets = torch.zeros(len(captions), max(lengths)).long()
  for i, cap in enumerate(captions):
    end = lengths[i]
    targets[i, :end] = cap[:end]
  return images, targets, lengths



def get_loader(method, vocab, batch_size):
  # train/validation paths
  # if method == 'train':
  #     root = 'data/train2014_resized'
  #     json = 'data/annotations/captions_train2014.json'
  # elif method =='val':
  #     root = 'data/val2014_resized'
  #     json = 'data/annotations/captions_val2014.json'

  # rasnet transformation/normalization
  transform = transforms.Compose([transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  print('get loader here')
  train_json = None
  with open('train_val_videodatainfo.json') as f:
    train_json = json.load(f)
  datadict = build_dicts(train_json,1)
  if method == 'val':
    datadict = build_dicts(train_json, None)

  data = DataLoader(ids=datadict, vocab=vocab, transform=transform)

  data_loader = torch.utils.data.DataLoader(dataset=data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate_fn)
  return data_loader



'''
Reads in train data npz file and the labels info json

Generates the following dictionaries:
- datapoint_id_dict: sen_id -> (sentence, video_id)
'''
def build_dicts(train_json, ids = None):
  # train_data = np.load(train_data_path)['a']
  train_data_ids = ['video1700', 'video1210', 'video1115', 'video1990',
  'video1701', 'video1211', 'video1116', 'video1891',
  'video1702', 'video1212', 'video1117', 'video1892',
  'video1703', 'video1213', 'video1118', 'video1893',
  'video1704', 'video1214', 'video1119', 'video1894',
  'video1705', 'video1215', 'video1120', 'video1895',
  'video1706', 'video1216', 'video1121', 'video1896',
  'video1707', 'video1217', 'video1122', 'video1897']
  if not ids:
    train_data_ids = ['video1801', 'video1802', 'video1803', 'video1804', 'video1805']
  print("LOOK HERE")
  print(train_data_ids)
  descriptions = train_json['sentences']
  datapoint_id_dict = {}
  i = 0
  for s in descriptions:
    if s['video_id'] in train_data_ids:
      # datapoint_id_dict[s['sen_id']] = [s['caption'], s['video_id']]
      datapoint_id_dict[i] = [s['caption'], s['video_id']]
      i += 1
  return datapoint_id_dict


def main():
  a = 3
  # print('starting data loader')
  # train_json = None
  # with open('train_val_videodatainfo.json') as f:
  # 	train_json = json.load(f)
  # datapoint_id_dict = build_dicts(train_json)
  # print(caption_to_id)

if __name__ == '__main__':
  main()