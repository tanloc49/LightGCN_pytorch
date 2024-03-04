import torch
from torch import nn


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, adj_matrix, embedding_size=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.adj_matrix = adj_matrix
        # Initial embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # Initialize embeddings to small random values
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self):
        # adj_matrix is the adjacency matrix of the user-item graph

        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)

        # Separate user and item embeddings after convolutions
        users, items = torch.split(embs, [self.num_users, self.num_items], 0)

        return users, items

    def getEmbedding(self, user_indices, item_indices_pos, item_indices_neg):
        users, items = self.forward()
        # Compute scores for positive and negative items
        user_indices_emb = users[user_indices]
        item_indices_pos_emb = items[item_indices_pos]
        item_indices_neg_emb = items[item_indices_neg]

        users_emb_ego = self.user_embedding(user_indices)
        pos_emb_ego = self.item_embedding(item_indices_pos)
        neg_emb_ego = self.item_embedding(item_indices_neg)
        return user_indices_emb, item_indices_pos_emb, item_indices_neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def predict(self, user_ids):
        users, items = self.forward()

        user_embedding = users[user_ids]
        all_items_embedding = items
        scores = torch.matmul(user_embedding, all_items_embedding.T)
        return scores

    def bpr_loss(self, user_indices, item_indices_pos, item_indices_neg):
        # Calculate BPR loss
        user_indices_emb, item_indices_pos_emb, item_indices_neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego = self.getEmbedding(
            user_indices, item_indices_pos, item_indices_neg)

        pos_scores = (user_indices_emb * item_indices_pos_emb).sum(dim=1)
        neg_scores = (user_indices_emb * item_indices_neg_emb).sum(dim=1)
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(user_indices))
        reg_loss = reg_loss * 1e-4
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + reg_loss
        return loss
