from tsfresh_feature_extraction import *

seed_everything(CFG.seed)

extracted_features_PCA=np.load("./numpy/extracted_features_PCA.npy")
test_feature_PCA=np.load("./numpy/test_feature_PCA.npy")

device=CFG.device
train_dataset_expend_encoded = AircraftDataset_expend_feature_extraction(df_train, train_label, torch.FloatTensor(extracted_features_PCA),True)
train_encoded_loader = DataLoader(
    train_dataset_expend_encoded,
    batch_size=CFG.batch_size,
    shuffle=True,
    drop_last=True,
)
test_dataset = AircraftDataset_no_expend_feature_extraction(df_train, test_label, torch.FloatTensor(test_feature_PCA))
test_encoded_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False)
# classifier_in = KMeans(n_clusters=CFG.num_in_feature_classes, random_state=CFG.seed).fit(get_input(test_dataset))
model = CustomModel(CFG.ts_input_dim,CFG.ts_hidden_dim).to(device)
loss_function = CustomLoss()

def fit_his_unit_in_unit_0(train_loader, test_loader_batch, in_unit_index, folder, RESUME=False):
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
    start_epoch = -1
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    if RESUME:
        path_checkpoint = folder + '/ckpt_unit_%s.pth' % (str(in_unit_index))  # 断点路径
        if os.path.isfile(path_checkpoint):
            checkpoint = torch.load(path_checkpoint, map_location=CFG.device)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            None

    for epoch in range(start_epoch + 1, CFG.epoches):
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
        # last_loss = loss
        if CFG.print_training_process and epoch % 10 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()},lr:{optimizer.state_dict()['param_groups'][0]['lr']}")
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir(folder):
                os.mkdir(folder)
            torch.save(checkpoint, folder + '/ckpt_unit_%s.pth' % (str(in_unit_index)))

        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder=folder)
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, f"model_in_unit_{in_unit_index}.pt",
                               cp_folder=folder)

def fit_his_unit(train_loader,path, RESUME=False):
    start_epoch=-1
    optimizer = getattr(torch.optim, CFG.optimizer)(model.parameters(), lr=CFG.lr)  # 优化器
    scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
    if RESUME:
        path_checkpoint = path+'/ckpt.pth'   # 断点路径
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
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(checkpoint,
                       path+'/ckpt.pth')
        if loss.mean() < CFG.jump_out_value:
            save_model_weights(model, "model_his.pt",
                               cp_folder=path)
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(model, "model_his.pt",
                               cp_folder=path)


if __name__ == "__main__":
    for idx, test_data in enumerate(test_encoded_loader):
        fit_his_unit_in_unit_0(train_encoded_loader, test_data, idx,
                                   "./model_checkpoints_ts/kernel without classify", RESUME=False)