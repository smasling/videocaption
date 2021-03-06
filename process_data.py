import os
import pickle
from collections import Counter
import nltk
# from PIL import Image
import json

class Vocabulary(object):
  def __init__(self):
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 0

  def add_word(self, word):
    if not word in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __call__(self, word):
    if not word in self.word2idx:
      return self.word2idx['<unk>']
    return self.word2idx[word]

  def __len__(self):
    return len(self.word2idx)

def build_vocab(train_json, threshold):
  counter = Counter()
  for s in train_json['sentences']:
    caption = s['caption']
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    counter.update(tokens)
  # omit non-frequent words
  words = [word for word, cnt in counter.items() if cnt >= threshold]

  vocab = Vocabulary()
  vocab.add_word('<pad>') # 0
  vocab.add_word('<start>') # 1
  vocab.add_word('<end>') # 2
  vocab.add_word('<unk>') # 3

  for i, word in enumerate(words):
    vocab.add_word(word)
  return vocab

# def resize_image(image):
#     width, height = image.size
#     if width > height:
#         left = (width - height) / 2
#         right = width - left
#         top = 0
#         bottom = height
#     else:
#         top = (height - width) / 2
#         bottom = height - top
#         left = 0
#         right = width
#     image = image.crop((left, top, right, bottom))
#     image = image.resize([224, 224], Image.ANTIALIAS)
#     return image

def main():
  print('process_data')
  caption_path = 'train_val_videodatainfo.json'
  vocab_path = './vocab.pkl'
  threshold = 5
  vocab = pickle.load(open(vocab_path, "rb"))
  # train_json = None
  # with open(caption_path) as f:
  # 	train_json = json.load(f)
  # vocab = build_vocab(train_json,threshold)
  # with open(vocab_path, 'wb') as f:
  # 	pickle.dump(vocab, f)

  # print("resizing images...")
  # splits = ['val','train']

  # for split in splits:
  #     folder = './data/%s2014' %split
  #     resized_folder = './data/%s2014_resized/' %split
  #     if not os.path.exists(resized_folder):
  #         os.makedirs(resized_folder)
  #     image_files = os.listdir(folder)
  #     num_images = len(image_files)
  #     for i, image_file in enumerate(image_files):
  #         with open(os.path.join(folder, image_file), 'r+b') as f:
  #             with Image.open(f) as image:
  #                 image = resize_image(image)
  #                 image.save(os.path.join(resized_folder, image_file), image.format)

  #   print("done resizing images...")


# print(vocab.__len__())
# print(vocab.word2idx.keys())
# main(caption_path, vocab_path, threshold)
