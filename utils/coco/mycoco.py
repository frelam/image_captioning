import json
from nltk import word_tokenize
import tqdm


def prepocessing():

    MAX_CAP_LEN = 5
    dir1 = '/media/f/新加卷1/train_split.0/sis/val.story-in-sequence.json'
    #############################################
    print("Filtering the captions by length...")
    with open(dir1) as f:
        l1 = json.load(f)
    keep_ann = {}
    keep_img = {}
    for ann in l1['annotations']:
        if len(word_tokenize(ann[0]['text'])) <= MAX_CAP_LEN:
            keep_ann[ann[0]['story_id']] = keep_ann.get(ann[0]['story_id'], 0) + 1
            keep_img[ann[0]['photo_flickr_id']] = keep_img.get(ann[0]['photo_flickr_id'], 0) + 1
    l1['annotations'] = \
        [ann for ann in l1['annotations'] \
         if keep_ann.get(ann[0]['story_id'], 0) > 0]
    l1['images'] = \
        [img for img in l1['images'] \
         if keep_img.get(img['id'], 0) > 0]
    ##############################################

    print("Filtering the captions by words...")
    keep_ann = {}
    keep_img = {}
    for ann in tqdm(l1['annotations']):
        keep_ann[ann['id']] = 1
        words_in_ann = word_tokenize(ann[0]['text'])
        for word in words_in_ann:
            if word not in vocab:
                keep_ann[ann[0]['story_id']] = 0
                break
        keep_img[ann[0]['photo_flickr_id']] = keep_img.get(ann[0]['photo_flickr_id'], 0) + 1

    l1['annotations'] = \
        [ann for ann in l1['annotations'] \
         if keep_ann.get(ann[0]['story_id'], 0) > 0]
    l1['images'] = \
        [img for img in l1['images'] \
         if keep_img.get(img['id'], 0) > 0]

    ##################################################
    story_id = {ann[0]['story_id']: [] for ann in l1['annotations']}
    for ann in l1['annotations']:
        story_id[ann[0]['story_id']] += [ann[0]['photo_flickr_id'], ann[0]['text']]


