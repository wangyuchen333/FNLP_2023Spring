import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ParsingModel(nn.Module):

    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ 
        Initialize the parser model. You can add arguments/settings as you want, depending on how you design your model.
        NOTE: You can load some pretrained embeddings here (If you are using any).
              Of course, if you are not planning to use pretrained embeddings, you don't need to do this.
        """
        super(ParsingModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))
        self.embed_to_hidden_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_features*self.embed_size,self.hidden_size)))
        self.embed_to_hidden_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1,self.hidden_size)))

        self.dropout = nn.Dropout(self.dropout_prob)

        self.hidden_to_logits_weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hidden_size,self.n_classes)))
        self.hidden_to_logits_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1,self.n_classes)))




    def forward(self, t):
        """
        Input: input tensor of tokens -> SHAPE (batch_size, n_features)
        Return: tensor of predictions (output after applying the layers of the network
                                 without applying softmax) -> SHAPE (batch_size, n_classes)
        """
        
        x = self.embeddings[t].view(t.shape[0],-1)
        h = F.relu(x.matmul(self.embed_to_hidden_weight)+self.embed_to_hidden_bias)
        h = self.dropout(h)
        logits = h.matmul(self.hidden_to_logits_weight)+self.hidden_to_logits_bias
        
        return logits
