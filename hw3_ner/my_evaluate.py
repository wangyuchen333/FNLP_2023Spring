import time
from collections import Counter

from model.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics
from evaluate.evaluate import get_score
import json

def dirty_data(data):
    label_m = {"NHCS":[],"NHVI":[],"NCSM":[],"NCGV":[],"NASI":[],"NT":[],"NS":[],"NO":[],"NATS":[],"NCSP":[]}

    # find the starting index of each entity
    start = -1
    last_label = None
    for i, tag in enumerate(data):
        if tag == 'O':
            last_label = None
            continue
        if tag == '<end>':
            continue
        pos, label = tag.split('-')
        if last_label is None:
            start = i
        if pos == 'E' or pos == 'S':
            # found the end of an entity, add it to the result list
            end = i + 1
            span = f"{start};{end}"
            label_m[label].append(span)
            last_label = None
        else:
            last_label = label
    new_label = []
    for label in label_m:
        nd=dict()
        nd['label']=label
        nd['span']=label_m[label]
        new_label.append(nd)
    return new_label

def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
            

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    # with open('output/prediction.json', 'w', encoding='utf-8') as f:
    #     f.write(metrics.report_scores())
    #     f.write(metrics.report_confusion_matrix())
    metrics.report_scores()
    metrics.report_confusion_matrix()
    with open('output/prediction.json', 'w', encoding='utf-8') as f:
        for i in range(len(pred_tag_lists)):
            data = {'id': i, 'entities': dirty_data(pred_tag_lists[i].copy())}
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data + '\n')

    with open('output/ground_truth.json', 'w', encoding='utf-8') as f:
        for i in range(len(test_tag_lists)):
            data = {'id': i, 'entities': dirty_data(test_tag_lists[i].copy())}
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data + '\n')
    scores = get_score('output/ground_truth.json', 'output/prediction.json')
    return pred_tag_lists, scores
