from os.path import join
from codecs import open

import json


def convert_to_bioes(data):
    text = data['context']
    entities = data['entities']
    words = []
    tags = []
    for i in range(len(text)):
        # Find the label of the current character, if any.
        label = 'O'
        for entity in entities:
            for sp in entity['span']:
                if entity['span']:
                    if i == int(sp.split(';')[0]):
                        label = f'B-{entity["label"]}'
                        # if len(entity['span']) > 1:
                        #     next_start = int(entity['span'][1].split(';')[0])
                        #     if next_start == i + 1:
                        #         label = f'M-{entity["label"]}'
                        if int(sp.split(';')[1])-int(sp.split(';')[0]) == 1:
                            label = f'S-{entity["label"]}'
                        break
                    elif i > int(sp.split(';')[0]) and i < int(sp.split(';')[-1])-1:
                        label = f'M-{entity["label"]}'
                        break
                    elif i == int(sp.split(';')[-1])-1:
                        label = f'E-{entity["label"]}'
                        break
        words.append(text[i])
        tags.append(label)
    return words, tags



def build_corpus(filename, make_vocab=True):
    """读取数据"""
    # assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    # with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
    #     word_list = []
    #     tag_list = []
    #     for line in f:
    #         if line != '\n':
    #             word, tag = line.strip('\n').split()
    #             word_list.append(word)
    #             tag_list.append(tag)
    #         else:
    #             word_lists.append(word_list)
    #             tag_lists.append(tag_list)
    #             word_list = []
    #             tag_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:  # read each line of the file
            data = json.loads(line.strip())
            word, tag = convert_to_bioes(data)
            word_lists.append(word)
            tag_lists.append(tag)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
