from datetime import datetime

from model import LightGCN
from process_data import *

import torch
from torch.utils.data import DataLoader


def evaluate_multiple_ks(model, test_loader, test_interactions, ks=None):
    if ks is None:
        ks = [5, 10, 20]
    model.eval()
    results = {k: {'precision': 0, 'recall': 0} for k in ks}
    total_users = 0

    with torch.no_grad():
        for user_ids in test_loader:
            user_ids = user_ids.to(model.user_embedding.weight.device)

            scores = model.predict(user_ids)
            for k in ks:
                _, top_k_indices = torch.topk(scores, k, dim=1)
                precisions, recalls, ndcgs = [], [], []

                for i, user_id in enumerate(user_ids):
                    true_items = test_interactions[user_id.item()]
                    pred_items = top_k_indices[i].tolist()

                    hit = np.isin(pred_items, true_items)
                    precision = np.mean(hit)
                    recall = np.sum(hit) / len(true_items) if true_items else 0

                    precisions.append(precision)
                    recalls.append(recall)

                results[k]['precision'] += np.sum(precisions)
                results[k]['recall'] += np.sum(recalls)

            total_users += user_ids.size(0)

    for k in ks:
        results[k]['precision'] /= total_users
        results[k]['recall'] /= total_users

    return results


def lightgcn_run():
    torch_sparse_tensor = create_matrix()

    model = LightGCN(num_users, num_items, torch_sparse_tensor, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 5

    # Train the model over specified epochs
    for epoch in range(num_epochs):
        samples = generate_samples(train_data, num_items, num_negatives=1)
        dataset = TrainDataset(samples)
        data_loader = DataLoader(dataset, batch_size=32768, shuffle=True)

        # Prepare DataLoader for testing
        test_dataset = TestDataset(unique_user_ids)
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
        model.train()
        total_loss = 0

        for user_indices, item_indices_pos, item_indices_neg in data_loader:
            optimizer.zero_grad()
            loss = model.bpr_loss(user_indices, item_indices_pos, item_indices_neg)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss}")

        ks = [5, 10, 20]
        results = evaluate_multiple_ks(model, test_loader, test_data, ks)
        for k in ks:
            print(
                f"      Top-{k} Results: Precision = {results[k]['precision']}, Recall = {results[k]['recall']}")

    formatted_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_state_dict = f'lightgcn_model_{formatted_date_time}.pth'

    torch.save(model.state_dict(), model_state_dict)

if __name__ == '__main__':
    lightgcn_run()

