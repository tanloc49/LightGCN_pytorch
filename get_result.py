from model import LightGCN
from process_data import *
from torch.utils.data import DataLoader


def get_result(model_file, unique_user_ids, k):
    torch_sparse_tensor = create_matrix()
    model = LightGCN(num_users, num_items, torch_sparse_tensor, 64)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    test_dataset = TestDataset(unique_user_ids)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for user_ids in test_loader:
            user_ids = user_ids.to(model.user_embedding.weight.device)
            scores = model.predict(user_ids)
            _, top_k_indices = torch.topk(scores, k, dim=1)
            results = {int(user_id): indices.tolist() for user_id, indices in zip(user_ids, top_k_indices)}

    return results

# results = get_result('lightgcn_model_2024-03-04_22-48-16.pth', [1,2,3,4,5], 5)
# print(results)
# ==>{1: [257, 496, 311, 313, 3027], 2: [608, 1271, 5303, 601, 566], 3: [608, 1271, 5303, 601, 566], 4: [632, 11776, 14127, 1728, 1782], 5: [2014, 786, 783, 2902, 710]}
