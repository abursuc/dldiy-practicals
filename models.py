import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_utils import ScaledEmbedding, ZeroEmbedding

class DotModel(nn.Module):
    
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim=32):
        
        super(DotModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)
        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)
                
        
    def forward(self, user_ids, item_ids):
        
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias

class DeepModel(nn.Module):
    
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim=30):
        
        super(DeepModel, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(2*embedding_dim,64)
        self.fcf = nn.Linear(64,1)
                
        
    def forward(self, user_ids, item_ids):
        
        user_embedding = self.user_embeddings(user_ids).squeeze()
        item_embedding = self.item_embeddings(item_ids).squeeze()
        x = torch.cat((user_embedding, item_embedding),1)#.squeeze()
        x = F.relu(self.fc1(F.dropout(x,0.3)))
        x = self.fcf(x)
        return x
