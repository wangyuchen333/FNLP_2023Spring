from datetime import datetime
import os
from torch import optim
import torch
from trainer import ParserTrainer
from parsing_model import ParsingModel
from parser_utils import load_and_preprocess_data
import json
from utils import evaluate
import numpy as np
import random


if __name__ == "__main__":
    random_seed = 123

    random.seed(random_seed)

    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # Note: Set debug to False, when training on entire corpus
    debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
    features = 36 if parser.unlabeled else 48
    classes = 3 if parser.unlabeled else 79
    if parser.unlabeled and debug:
        classes = 77
    parser.model = ParsingModel(embeddings,n_features=features,n_classes=classes) # You can add more arguments, depending on how you designed your parsing model

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### TODO:
    ###     1. Call an optimizer (no need to specify parameters yet, which will be implemented during training)
    ###     2. Construct the Cross Entropy Loss Function in variable `loss_func`
    optimizer = optim.Adam(params=parser.model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    trainer = ParserTrainer(
        train_data=train_data,
        dev_data=dev_data,
        optimizer=optimizer,
        loss_func=loss_func,
        output_path=output_path,
        batch_size=1024,
        n_epochs=30,
        lr=0.0001,
    )
    trainer.train(parser, train_data, dev_data, output_path)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model = ParsingModel(embeddings,n_features=features,n_classes=classes)
        parser.model.load_state_dict(torch.load(output_path))
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
        uas,las = evaluate(all_head, all_ex_head)  # To check the format of the input, please refer to the utils.py
        for i, ex in enumerate(test_data[:2]):  # 仅展示前两个句子的解析结果
            head = [-1] * len(ex['word'])
            for dependency in dependencies[i]:
                h, t, label = dependency
                head[t] = [h, label]
            all_head.append(head[1:]) 
            all_ex_head.append(list(zip(ex['head'][1:], ex['label'][1:])))
            sentence = " ".join([parser.id2tok[w] for w in ex['word']])
            print("Sentence:", sentence)
            print("Dependencies:", head[1:])  # 打印解析结果
            print()
        print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
        print("Done!")
