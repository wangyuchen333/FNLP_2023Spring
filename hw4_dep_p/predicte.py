import os
import torch
from parsing_model import ParsingModel
import json
from utils import evaluate
import numpy as np
import random
from parser_utils import read_conll, Parser

if __name__ == "__main__":
    random_seed = 123

    random.seed(random_seed)

    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    
    
    print("Loading data...",)
    train_set = read_conll(os.path.join('./data', 'train.conll'),
                        lowercase=True)
    test_set = read_conll(os.path.join('./data', 'real_test.conll'),
                          lowercase=True)
    parser = Parser(train_set)
    test_data = parser.vectorize(test_set)

    features = 36 if parser.unlabeled else 48
    classes = 3 if parser.unlabeled else 79
    word_vectors = {}
    for line in open('./data/en-cw.txt').readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    parser.model = ParsingModel(embeddings_matrix,n_features=features,n_classes=classes)
    parser.model.load_state_dict(torch.load("results/model.weights"))
    print("Final evaluation on test set",)
    parser.model.eval()
    _, dependencies = parser.parse(test_data)
    all_head = []
    all_ex_head = []
    for i, ex in enumerate(test_data):
        head = [-1] * len(ex['word'])
        for dependency in dependencies[i]:
            h, t, label = dependency
            head[t] = [h, label]
        ex_label = [parser.id2tok[w].replace('<l>:', '') for w in ex['label'][1:]]
        all_head.append(head[1:]) 
        all_ex_head.append(list(zip(ex['head'][1:],ex_label)))
    with open('./prediction.json', 'w') as fh:
        json.dump(dependencies, fh)
    uas,las = evaluate(all_head, all_ex_head) 
    # for i, ex in enumerate(test_data[:2]):
    #     head = [-1] * len(ex['word'])
    #     for dependency in dependencies[i]:
    #         h, t, label = dependency
    #         head[t] = [h, label]
    #     ex_label = [parser.id2tok[w].replace('<l>:', '') for w in ex['label'][1:]]
    #     all_head.append(head[1:]) 
    #     all_ex_head.append(list(zip(ex['head'][1:], ex['label'][1:])))
    #     sentence = " ".join([parser.id2tok[w] for w in ex['word']])
    #     print("Sentence:", sentence)
    #     print("Dependencies:", list(zip(ex['head'][1:], ex_label)))
    #     print("Predicted dependencies:", head[1:])
    #     print()
    print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
    print("Done!")