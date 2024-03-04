import os

import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix, diags
from torch.utils.data import Dataset

train_file_path = 'dataset/train.txt'
test_file_path = 'dataset/test.txt'

train_data = {}
test_data = {}
max_user_id = -1
max_item_id = -1
unique_user_ids = []

with open(train_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if parts:
            user_id = int(parts[0])
            item_ids = [int(item) for item in parts[1:]]
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            train_data[user_id] = item_ids
            unique_user_ids.append(user_id)

with open(test_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if parts:
            user_id = int(parts[0])
            item_ids = [int(item) for item in parts[1:]]
            max_item_id = max(max_item_id, max(item_ids))
            test_data[user_id] = item_ids

num_users = max_user_id + 1
num_items = max_item_id + 1

def generate_samples(data, total_items, num_negatives=4):
    samples = []
    for user, positive_items in data.items():
        for train_users in range(len(positive_items)):
            positive_item = np.random.choice(positive_items)
            for _ in range(num_negatives):
                neg_item = np.random.randint(0, total_items)
                while neg_item in positive_items:
                    neg_item = np.random.randint(0, total_items)
                samples.append((user, positive_item, neg_item))
    return samples

class TrainDataset(Dataset):
    def __init__(self, samples):
        super(TrainDataset, self).__init__()
        self.users, self.pos_items, self.neg_items = zip(*samples)
        self.users = torch.tensor(self.users, dtype=torch.long)
        self.pos_items = torch.tensor(self.pos_items, dtype=torch.long)
        self.neg_items = torch.tensor(self.neg_items, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]

class TestDataset(Dataset):
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx]


def create_matrix():
    file_path = os.path.join("matrices", "interaction_matrix")
    if os.path.exists(file_path):
        tensor = torch.load(file_path)
        return tensor
    else:
        train_items = set([item for sublist in train_data.values() for item in sublist])
        item_to_index = {item_id: index for index, item_id in enumerate(sorted(train_items))}
        train_interaction_data = []
        row_indices = []
        col_indices = []
        for user_id, item_ids in train_data.items():
            for item_id in item_ids:
                row_indices.append(user_id)
                col_indices.append(item_to_index[item_id])
                train_interaction_data.append(1)

        interaction_matrix = csr_matrix((train_interaction_data, (row_indices, col_indices)),
                                        shape=(num_users, num_items))
        n = num_users + num_items
        adj_matrix = lil_matrix((n, n))
        adj_matrix[:num_users, num_users:] = interaction_matrix
        adj_matrix[num_users:, :num_users] = interaction_matrix.T
        adj_matrix = adj_matrix.tocsr()
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        A_normalized = D_inv_sqrt.dot(adj_matrix).dot(D_inv_sqrt)
        return _convert_sp_mat_to_sp_tensor(A_normalized, "interaction_matrix")


def _convert_sp_mat_to_sp_tensor(X, filename):
    if not os.path.exists("matrices"):
        os.makedirs("matrices")
    file_path = os.path.join("matrices", filename)

    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    tensor = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    torch.save(tensor, file_path)
    return tensor


