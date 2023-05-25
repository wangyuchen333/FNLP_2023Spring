import argparse
# import torch.nn as nn
import json
from sklearn.model_selection import train_test_split
from data import build_corpus,build_map
from utils import extend_maps, prepocess_data_for_lstmcrf
from my_evaluate import bilstm_train_and_eval
import pickle

# def load_data(train_data_path, test_data_path):
#     """
#     Load, split and pack your data.
#     Input: The paths to the files of your original data
#     Output: Packed data, e.g., a list like [train_data, dev_data, test_data]
#     """
#     with open(train_data_path, 'r') as f:
#         train_data = json.load(f)
    
#     dev_data = None # 如果没有单独的开发集，则将其设置为None
    
#     with open(test_data_path, 'r') as f:
#         test_data = json.load(f)

#     return [train_data, dev_data, test_data]


# class Model(nn.Module):
#     def __init__(self, ): # You can add more arguments
#         pass
#     def forward(): # You can add more arguments
#         """
#         The implementation of NN forward function.
#         Input: The data of your batch_size
#         Output: The result tensor of this batch
#         """
#         pass


# class Trainer():
#     def __init__(self, model: Model, ): # You can add more arguments
#         self.model = model
#         pass
#     def train(train_data, dev_data, ): # You can add more arguments
#         """
#         Given packed train_data, train the model (including optimization),
#         save checkpoints, print loss, log the best epoch, and run tests on packed dev_data
#         """
#         pass
#     def test(data, mode, ): # You can add more arguments
#         """
#         Given packed data, run the model and predict results
#         This function should be able to load a model from a checkpoint

#         """
#         if mode == 'dev_eval':
#             pass # Directly run tests on dev_data and print results in the console
#         elif mode == 'test_eval':
#             pass # Here you should save the results to ./output/output.json
#         else:
#             pass


def main():
    # NOTE: You can use variables in args as further arguments of the following functions
    train_data_path = './input/train_data.json'
    test_data_path = './input/test_data.json'
    # train_data, dev_data, test_data = load_data(train_data_path, test_data_path)
    # model = Model()
    # trainer = Trainer(model, )
    # trainer.train(train_data, dev_data, )
    # trainer.test(test_data, mode='test_eval')

    word_lists, tag_lists = build_corpus(train_data_path,make_vocab=False)
    data = list(zip(word_lists, tag_lists))
    train_data, dev_data = train_test_split(data, test_size=0.1, random_state=42)
    # dev_data,test_data = train_test_split(dev_data, test_size=0.8, random_state=42)
    test_data = dev_data
    # dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    # test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    train_word_lists, train_tag_lists = zip(*train_data)
    word2id = build_map(train_word_lists)
    tag2id = build_map(train_tag_lists)
    dev_word_lists, dev_tag_lists = zip(*dev_data)
    test_word_lists, test_tag_lists = zip(*test_data)

    print("正在训练评估Bi-LSTM+CRF模型...")

    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    with open('./ckpts/crf_word2id.pkl', 'wb') as f:
        pickle.dump(crf_word2id, f)
    
    with open('./ckpts/crf_tag2id.pkl', 'wb') as f:
        pickle.dump(crf_tag2id, f)
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, scores = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )
    print(scores)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="arguments")
    # # You can add more arguments as you want
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=1,
    #     help="Batch Size"
    # )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=20,
    #     help="Epochs"
    # )
    # args = parser.parse_args()
    main()