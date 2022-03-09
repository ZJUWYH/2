from utilities import *
from config import *
from data_preparing import *
from DNN_model import *

seed_everything(CFG.seed)
DATA_PATH = "./Data_FD003/preprocessed data/"
attribute = ['Unit', 'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
             'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
df_train = pd.read_csv(DATA_PATH + 'TD_data.csv', names=attribute, header=None)
df_test = pd.read_csv(DATA_PATH + 'Test_data.csv', names=attribute, header=None)

df_train = Preprocessing.add_timeseries(df_train)
df_test = Preprocessing.add_timeseries(df_test)

train_data_expend = AircraftDataset_expend(df_train, False)  # 不插0计算
test_dataset = AircraftDataset(df_test)

train_data_expend_norul = AircraftDataset_expend_norul(df_train)

train_data_expend_loader = DataLoader(train_data_expend_norul,
                                      batch_size=CFG.ae_batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=collate_fn)

aemodel = GRUAutoEncoder().to(CFG.device)
loss_func = nn.MSELoss()
device = CFG.device


def train_ae(train_loader, RESUME=False):
    start_epoch = -1
    optimizer = getattr(torch.optim, CFG.optimizer)(aemodel.parameters(), lr=CFG.ae_lr)
    # scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(optimizer, gamma=CFG.sc_Gamma)  # 指数型学习率
    for layer in aemodel.modules():
        if isinstance(layer, nn.GRU):
            nn.init.xavier_uniform_(layer.weight_hh_l0.data, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.xavier_uniform_(layer.weight_ih_l0.data, gain=nn.init.calculate_gain('sigmoid'))
    if RESUME:
        path_checkpoint = './model_checkpoints_ae/ae/ckpt.pth'  # 断点路径
        if os.path.isfile(path_checkpoint):
            checkpoint = torch.load(path_checkpoint, map_location=CFG.device)  # 加载断点

            aemodel.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            # scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch + 1, CFG.epoches):
        for data in train_loader:
            decoded = aemodel(data.to(CFG.device))[0]
            loss = loss_func(decoded.data, data.to(CFG.device).data)
            # loss = loss_function(pred_batch, target_batch, train_data, test_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        # scheduler.step(loss,last_loss)
        # last_loss = loss
        if CFG.print_training_process and epoch % 10 == 0:
            print(f"epoch:{epoch}, loss:{loss.item()},lr:{optimizer.state_dict()['param_groups'][0]['lr']}")
            checkpoint = {
                "net": aemodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                # 'scheduler': scheduler.state_dict()
            }
            if not os.path.isdir("./model_checkpoints_ae/ae"):
                os.mkdir("./model_checkpoints_ae/ae")
            torch.save(checkpoint,
                       './model_checkpoints_ae/ae/ckpt.pth')
        if loss.mean() < 0.035:
            save_model_weights(aemodel, "model_ae.pt",
                               cp_folder="./model_checkpoints_ae/ae")
            break
        elif epoch == (CFG.epoches - 1):
            print(f"epoch:{epoch}, loss:{loss.item()}")
            save_model_weights(aemodel, "model_ae.pt",
                               cp_folder="./model_checkpoints_ae/ae")


####encoder之后进入dnn模型

# feature_list_expend=encode_feature_extraction(train_data_expend)#获得切片后测试集的featurelist
# feature_list_test=encode_feature_extraction(test_dataset)#获得testdata的featurelist

if __name__ == "__main__":
    train_encode = True
    extract_feature= False
    if train_encode:
        train_ae(train_data_expend_loader, RESUME=True)
    elif extract_feature:
        feature_list_expend = encode_feature_extraction(train_data_expend)  # 获得切片后测试集的featurelist
        feature_list_test = encode_feature_extraction(test_dataset)  # 获得testdata的featurelist
