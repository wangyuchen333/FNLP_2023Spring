from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
import pickle
from my_evaluate import dirty_data
import json
from evaluate.evaluate import get_score
BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

def get_id(filename):
    id_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:  # read each line of the file
            data = json.loads(line.strip())
            id = data['id']
            id_list.append(id)
    return id_list

def main():
    print("读取数据...")
    # train_word_lists, train_tag_lists, word2id, tag2id = \
    #     build_corpus()
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus('./input/test_data.json', make_vocab=False)
    id_list = get_id('./input/test_data.json')

    print("加载并评估bilstm+crf模型...")
    # crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    with open('./ckpts/crf_word2id.pkl', 'rb') as f:
        crf_word2id = pickle.load(f)
    
    with open('./ckpts/crf_tag2id.pkl', 'rb') as f:
        crf_tag2id = pickle.load(f)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    # metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()
    with open('output/output.json', 'w', encoding='utf-8') as f:
        for i in range(len(lstmcrf_pred)):
            data = {'id': id_list[i], 'entities': dirty_data(lstmcrf_pred[i].copy())}
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data + '\n')

    # with open('output/ground_truth.json', 'w', encoding='utf-8') as f:
    #     for i in range(len(target_tag_list)):
    #         data = {'id': i, 'entities': dirty_data(target_tag_list[i].copy())}
    #         json_data = json.dumps(data, ensure_ascii=False)
    #         f.write(json_data + '\n')
    # scores = get_score('output/ground_truth.json', 'output/prediction.json')
    # print(scores)

if __name__ == "__main__":
    main()
