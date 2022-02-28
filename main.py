from utilities import *
from config import *
from diwang_feature_exrtaction import *
from data_preparing import *
from DNN_model import *

# feature_extraction v1 with DNN method
# set random seed
seed_everything(CFG.seed)
device = CFG.device  # .to(device)

# Initialize the model
model = CustomModel().to(device)
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

# Instantiate training and test sets
train_dataset = TrainingFeature(G, dev_G, dev2_G, Tal0)
test_dataset = TestingFeature(tG_delete[0:2],
                              dev_tG_delete[0:2],
                              dev2_tG_delete[0:2],
                              RUL_delete[0:2])

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    drop_last=True,
    )
    #pin_memory=True,
    #worker_init_fn=worker_init_fn  # train_dataset

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False)  # test_dataset

loss_function = CustomLoss()  # identical loss function

optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
#scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
scheduler= myscheduler(optimizer,CFG.decay)


def fit_his_unit_in_unit_0(train_loader, test_loader_batch, in_unit_index):
    """
    传入 in_unit 的feature来获得kernel完成预测
    :param train_loader: 含有多个batch, 是字典形式的
    :param test_loader_batch: test loader 的一个batch， 是字典形式的
    :param in_unit_index: 记录in unit的index
    :return: None
    """
    for epoch in range(CFG.epoches):
        for data in train_loader:
            pred_batch = model(data["input"].to(device))
            target_batch = data["RUL"].unsqueeze(-1).to(device)
            train_data = data["input"].to(device)
            test_data = test_loader_batch["input"].to(device)
        loss = loss_function(pred_batch, target_batch, train_data, test_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        scheduler.step(loss)
        if CFG.print_training_process and epoch % 100 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()}")
        if loss.mean() < 10:
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder=".\\model_checkpoints")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder=".\\model_checkpoints")


def pred_in_units(train_loader, test_loader):
    n_in_units = len(test_loader)
    pred_in_RUL = np.zeros(n_in_units, dtype=float)
    for idx, test_data in enumerate(test_loader):
        fit_his_unit_in_unit_0(train_loader, test_data, idx)
        pred_in_RUL[idx] = model(test_data["input"].to(device))
    return pred_in_RUL


print(pred_in_units(train_loader, test_loader), RUL_delete[0:2])
