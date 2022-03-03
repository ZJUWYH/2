from main import *


n_in_units = len(test_loader)
pred_in_RUL = np.zeros(n_in_units, dtype=float)
for idx, test_data in enumerate(test_loader):
    # if idx==8: break
    path = ".\\model_checkpoints" + f"\\model_in_unit_{idx}.pt"
    model.load_state_dict(torch.load(path))
    pred_in_RUL[idx] = model(test_data["input"].to(device)).item()

print(pred_in_RUL)

