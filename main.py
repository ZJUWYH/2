from utilities import *
from config import *
from diwang_feature_exrtaction import *
from data_preparing import *
from DNN_model import *
from transformers import get_linear_schedule_with_warmup

# feature_extraction v1 with DNN method
# set random seed
seed_everything(CFG.seed)
device = CFG.device  # .to(device)

# build classifier
tGG = fusion_tG(tG_delete, dev_tG_delete, dev2_tG_delete)
classifier_in = KMeans(n_clusters=CFG.num_in_feature_classes, random_state=CFG.seed).fit(tGG)
classifier_his= KMeans(n_clusters=CFG.num_in_feature_classes,
                       random_state=CFG.seed).fit(TrainingFeature(G, dev_G, dev2_G, Tal0).input)

# Initialize the model
model = CustomModel(CFG.input_feature).to(device)
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

# Instantiate training and test sets
train_dataset = TrainingFeature(G, dev_G, dev2_G, Tal0)
train_classified_dataset=Classified_mean_train_features(G, dev_G, dev2_G, Tal0,classifier_his)

test_dataset = TestingFeature(tG_delete[0:2],
                              dev_tG_delete[0:2],
                              dev2_tG_delete[0:2],
                              RUL_delete[0:2],
                              classifier_in)

test_dataset_compelete = TestingFeature(tG_delete,
                              dev_tG_delete,
                              dev2_tG_delete,
                              RUL_delete,
                              classifier_in)

test_classified_dataset = Classified_mean_test_features(tG_delete,
                                                        dev_tG_delete,
                                                        dev2_tG_delete,
                                                        RUL_delete, classifier_in)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    drop_last=True,
)
# pin_memory=True,
# worker_init_fn=worker_init_fn  # train_dataset

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False)  # test_dataset 只选了两个作为测试的

test_loader_compelete = DataLoader(
    test_dataset_compelete,
    batch_size=1,
    shuffle=False)  # test_dataset 完整的，其中有label

test_classified_loader = DataLoader(
    test_classified_dataset,
    batch_size=1,
    shuffle=False)  # test_dataset_classified num_class batch的数量为类的个数

train_classified_loader=DataLoader(
    train_classified_dataset,
    batch_size=1,
    shuffle=False)  # train_dataset_classified num_class batch 的数量为类的个数

loss_function = CustomLoss()  # identical loss function

# optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
# scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# scheduler= myscheduler(optimizer,CFG.decay)


def fit_his_unit_in_unit_0(train_loader, test_loader_batch, in_unit_index, RESUME=False):
    """
    传入 in_unit 的feature来获得kernel完成预测
    :param train_loader: 含有多个batch, 是字典形式的
    :param test_loader_batch: test loader 的一个batch， 是字典形式的
    :param in_unit_index: 记录in unit的index
    :return: None
    """
    # last_loss = torch.tensor([1e5], dtype=torch.float).to(CFG.device)
    optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
    scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    start_epoch=-1
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    if RESUME:
        path_checkpoint = './model_checkpoints/kernel without classify/ckpt_unit_%s.pth' % (str(in_unit_index))  # 断点路径
        if os.path.isfile(path_checkpoint):
            checkpoint = torch.load(path_checkpoint,map_location=CFG.device)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            None

    for epoch in range(start_epoch+1,CFG.epoches):
        for data in train_loader:
            pred_batch = model(data["input"].to(device))
            target_batch = data["RUL"].unsqueeze(-1).to(device)
            train_data = data["input"].to(device)
            test_data = test_loader_batch["input"].to(device)
            loss = loss_function(pred_batch, target_batch, train_data, test_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # scheduler.step(loss,last_loss)
        last_loss = loss
        if CFG.print_training_process and epoch % 100 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()}")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir("./model_checkpoints/kernel without classify"):
                os.mkdir("./model_checkpoints/kernel without classify")
            torch.save(checkpoint, './model_checkpoints/kernel without classify/ckpt_unit_%s.pth' % (str(in_unit_index)))

        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder="./model_checkpoints/kernel without classify")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder="./model_checkpoints/kernel without classify")


def fit_his_unit_in_unit_classified(train_loader, test_loader_classified_batch, labels_index,RESUME=False):
    """
    :param train_loader: 含有多个batch, 是字典形式的
    :param test_loader_classified_batch: test_loader_classified 的一个batch， 是字典形式的
    :param labels_index: label 的 index
    :return:
    """
    start_epoch=-1
    optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
    scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    if RESUME:
        path_checkpoint = './model_checkpoints/kernel with classify/ckpt_cls_%s.pth' % (str(labels_index))  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch+1,CFG.epoches):
        for data in train_loader:
            pred_batch = model(data["input"].to(device))
            target_batch = data["RUL"].unsqueeze(-1).to(device)
            train_data = data["input"].to(device)
            test_data = test_loader_classified_batch["input"].to(device)
            loss = loss_function(pred_batch, target_batch, train_data, test_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # scheduler.step(loss,last_loss)
        last_loss = loss
        if CFG.print_training_process and epoch % 100 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()}")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir("./model_checkpoints/kernel with classify"):
                os.mkdir("./model_checkpoints/kernel with classify")
            torch.save(checkpoint,
                       './model_checkpoints/kernel with classify/ckpt_cls_%s.pth' % (str(labels_index)))
        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, f"model_in_cls_{labels_index}.pt",
                               cp_folder="./model_checkpoints/kernel with classify")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, f"model_in_cls_{labels_index}.pt",
                               cp_folder="./model_checkpoints/kernel with classify")

def fit_his_unit_classified(train_loader, train_loader_classified_batch, labels_index,RESUME=False):
    """
    :param train_loader: 含有多个batch, 是字典形式的
    :param train_loader_classified_batch: train_loader_classified 的一个batch， 是字典形式的
    :param labels_index: label 的 index
    :return:
    """
    start_epoch=-1
    optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
    scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    if RESUME:
        path_checkpoint = './model_checkpoints/kernel with classify train/ckpt_cls_%s.pth' % (str(labels_index))  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch+1,CFG.epoches):
        for data in train_loader:
            pred_batch = model(data["input"].to(device))
            target_batch = data["RUL"].unsqueeze(-1).to(device)
            train_data = data["input"].to(device)
            test_data = train_loader_classified_batch["input"].to(device)
            loss = loss_function(pred_batch, target_batch, train_data, test_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # scheduler.step(loss,last_loss)
        last_loss = loss
        if CFG.print_training_process and epoch % 100 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()}")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir("./model_checkpoints/kernel with classify train"):
                os.mkdir("./model_checkpoints/kernel with classify train")
            torch.save(checkpoint,
                       './model_checkpoints/kernel with classify train/ckpt_cls_%s.pth' % (str(labels_index)))
        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, f"model_his_cls_{labels_index}.pt",
                               cp_folder="./model_checkpoints/kernel with classify train")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, f"model_his_cls_{labels_index}.pt",
                               cp_folder="./model_checkpoints/kernel with classify train")


def fit_his_unit(train_loader,RESUME=False):
    start_epoch=-1
    optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
    scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
    if RESUME:
        path_checkpoint = './model_checkpoints/no kernel no classify/ckpt.pth'   # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['scheduler'])


    for epoch in range(start_epoch+1,CFG.epoches):
        for data in train_loader:
            pred_batch = model(data["input"].to(device))
            target_batch = data["RUL"].unsqueeze(-1).to(device)
            loss = F.mse_loss(pred_batch, target_batch.float())
        # loss = loss_function(pred_batch, target_batch, train_data, test_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # scheduler.step(loss,last_loss)
        # last_loss = loss
        if CFG.print_training_process and epoch % 100 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()},lr:{optimizer.state_dict()['param_groups'][0]['lr']}")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir("./model_checkpoints/no kernel no classify"):
                os.mkdir("./model_checkpoints/no kernel no classify")
            torch.save(checkpoint,
                       './model_checkpoints/no kernel no classify/ckpt.pth')
        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, "model_his.pt",
                               cp_folder="./model_checkpoints/no kernel no classify")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, "model_his.pt",
                               cp_folder="./model_checkpoints/no kernel no classify")


def pred_in_units(train_loader, test_loader, RESUME=False):
    """
    :param train_loader:
    :param test_loader: loader about features /features mean from train data or test data
    :return: required model
    """
    n_in_units = len(test_loader)
    # pred_in_RUL = np.zeros(n_in_units, dtype=float)
    for idx, test_data in enumerate(test_loader):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
        fit_his_unit_in_unit_0(train_loader, test_data, idx)
        # pred_in_RUL[idx] = model(test_data["input"].to(device))


if __name__ == "__main__":
    fit_his_unit(train_loader,RESUME=True)
    for idx, test_data in enumerate(test_loader):
        fit_his_unit_in_unit_0(train_loader, test_data, idx, RESUME=True)
    for idx, test_classified_data in enumerate(test_classified_loader):
        fit_his_unit_in_unit_classified(train_loader, test_classified_data, idx, RESUME=True)
    for idx, train_classified_data in enumerate(train_classified_loader):
        fit_his_unit_classified(train_loader, train_classified_data, idx, RESUME=True)

