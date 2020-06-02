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
    fle = "featuresFolder/" + movie + ".npy"
    buf = np.load(fle)

    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    target = torch.Tensor(caption)
    return torch.from_numpy(buf[0]).type(torch.FloatTensor), target

  def __len__(self):
    return len(self.ids)


def collate_fn(data):
  images, captions = zip(*data)
  S, _, H, C = images[0].shape
  mx = 0
  for i in range(len(images)):
    f = images[i].shape[1]
    if f > mx:
      mx = f
  newImages = torch.zeros(len(images), S, mx, H, C)
  #print(newImages.shape)
  for i, image in enumerate(images):
      #print(image.shape)
      newImages[i, :,  :image.shape[1]] = image
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
  datadict = build_dicts(train_json,0)
  if method == 'val':
    datadict = build_dicts(train_json, 1)

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
  movies = [filename[:-4] for filename in os.listdir('testFolder')]
  descriptions = train_json['sentences']
  if ids == 0:
    train_data_ids = movies[:(int)((0.8) * len(movies))]
  else:
    train_data_ids = movies[(int)((0.8) * len(movies)):]

  datapoint_id_dict = {}
  i = 0
  for s in descriptions:
    if s['video_id'] in train_data_ids:
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
