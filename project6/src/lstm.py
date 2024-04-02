import torch
from torch.autograd import forward_ad
import torch.nn as nn 

class LSTM(nn.Module):
    """Long Short Term Memory (LSTM)"""
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )

        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )

        self.out = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)

        ave_pool = torch.mean(x, 1)
        max_pool = torch.max(x, 1)

        out = torch.cat((ave_pool, max_pool), 1)

        out = self.out(out)
        return out 