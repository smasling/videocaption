import json
from os import listdir


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
