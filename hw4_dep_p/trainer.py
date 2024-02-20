import torch
import torch.nn as nn
from torch import optim
from utils import evaluate
from tqdm import tqdm
import math
from parser_utils import minibatches, AverageMeter

class ParserTrainer():

    def __init__(
        self,
        train_data,
        dev_data,
        optimizer,
        loss_func,
        output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005, 
    ): # You can add more arguments
        """
        Initialize the trainer.
        
        Inputs:
            - train_data: Packed train data
            - dev_data: Packed dev data
            - optimizer: The optimizer used to optimize the parsing model
            - loss_func: The cross entropy function to calculate loss, initialized beforehand
            - output_path (str): Path to which model weights and results are written
            - batch_size (int): Number of examples in a single batch
            - n_epochs (int): Number of training epochs
            - lr (float): Learning rate
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.output_path = output_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        ### TODO: You can add more initializations here


    def train(self, parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005): # You can add more arguments as you need
        """
        Given packed train_data, train the neural dependency parser (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        """
        best_dev_LAS = 0

        ### TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize
        self.optimizer = optim.Adam(params=parser.model.parameters(),lr=self.lr)
        self.loss_func = loss_func = nn.CrossEntropyLoss()

        for epoch in range(self.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epochs))
            dev_UAS, dev_LAS = self._train_for_epoch(parser, train_data, dev_data, self.optimizer, self.loss_func, self.batch_size)
            # TODO: you can change this part, to use either uas or las to select best model
            if dev_LAS > best_dev_LAS:
                best_dev_LAS = dev_LAS
                print("New best dev LAS! Saving model.")
                torch.save(parser.model.state_dict(), self.output_path)
            print("")


    def _train_for_epoch(self, parser, train_data, dev_data, optimizer, loss_func, batch_size): # You can add more arguments as you need
        """ 
        Train the neural dependency parser for single epoch.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        Return:
            - dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
        """
        parser.model.train() # Places model in "train" mode, e.g., apply dropout layer, etc.
        ### TODO: Train all batches of train_data in an epoch.
        ### Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)

        n_minibatches = math.ceil(len(train_data) / batch_size)
        loss_meter = AverageMeter()
        with tqdm(total=n_minibatches) as prog:
            size = 3 if parser.unlabeled else 79
            for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size,size)):
                optimizer.zero_grad()
                loss = 0.
                train_x = torch.from_numpy(train_x).long()
                train_y = torch.from_numpy(train_y.nonzero()[1]).long()
                logits = parser.model.forward(train_x)
                loss = loss_func(logits,train_y)
                loss.backward()
                optimizer.step()
                prog.update(1)
                loss_meter.update(loss.item())
        print ("Average Train Loss: {}".format(loss_meter.avg))
        print("Evaluating on dev set",)
        parser.model.eval() # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        _,dependencies = parser.parse(self.dev_data)
        all_head = []
        all_ex_head = []
        for i, ex in enumerate(self.dev_data):
            head = [-1] * len(ex['word'])
            for dependency in dependencies[i]:
                h, t, label = dependency
                head[t] = [h, label]
            ex_label = [parser.id2tok[w].replace('<l>:', '') for w in ex['label'][1:]]
            all_head.append(head[1:]) 
            all_ex_head.append(list(zip(ex['head'][1:],ex_label)))
        uas,las = evaluate(all_head, all_ex_head)  # To check the format of the input, please refer to the utils.py
        print("- dev UAS: {:.2f}".format(uas * 100.0), "- dev LAS: {:.2f}".format(las * 100.0))
        return uas, las