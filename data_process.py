from config import *

DATA_PATH = "./Dataset 1/preprocessed data/"

attribute = ['Unit', 'T24', 'T50', 'P30', 'Nf', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']

df_train = pd.read_csv(DATA_PATH + 'TD_data.csv', names=attribute, header=None)
df_test = pd.read_csv(DATA_PATH + 'Test_data.csv', names=attribute, header=None)

df_ = df_train[df_train['Unit'] < 5].reset_index(drop=True)


class Preprocessing:

    def add_timeseries(df):
        df0 = df.copy()
        df0["Time"] = df0.groupby(["Unit"]).cumcount() + 1
        return df0


# df_=Preprocessing.add_timeseries(df_)

class AircraftDataset(Dataset):
    def __init__(self, df):
        self.df = df.groupby("Unit").agg(list).reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        data = {}
        sensor = ['T24', 'T50', 'P30', 'Nf', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32']
        multi_sensor = []
        for sensor_name in sensor:
            multi_sensor.append(np.array(self.df[sensor_name].values.tolist()[idx]))
            single_sensor = np.array(self.df[sensor_name].values.tolist()[idx])[:, None]
            data[sensor_name] = torch.tensor(single_sensor, dtype=torch.float)
        multi_sensor = np.vstack(multi_sensor).transpose(1, 0)
        data["input"] = torch.tensor(multi_sensor, dtype=torch.float)
        data["lifetime"] = torch.tensor(len(multi_sensor), dtype=torch.int64)
        data["timeseries"] = torch.tensor(np.array(self.df["Time"].values.tolist()[idx])[:, None], dtype=torch.int64)

        return data


def feature_extration_from_his_unit_v1(data):  # data为类AircraftDataset的一个实例
    input_dim = CFG.input_dim  # 11
    feature = np.zeros((len(data), 3, input_dim))
    coefs = np.zeros((len(data), input_dim, 3))
    for unit in range(len(data)):
        data_l = data[unit]["input"].numpy()
        timeseries_l = data[unit]["timeseries"].numpy()
        feature_l = np.zeros((3, input_dim))
        for sensor in range(data_l.shape[1]):
            coef = np.polyfit(timeseries_l.reshape(-1), data_l[:, sensor], 2)
            tau = timeseries_l[-1]
            feature_l[0][sensor] = np.polyval(coef, tau)
            feature_l[1][sensor] = 2 * coef[0] * tau + coef[1]
            feature_l[2][sensor] = 2 * coef[0]
            coefs[unit][sensor] = coef
        feature[unit] = feature_l
    return feature, coefs


def feature_fusion_for_in_unit_v1(his_data, in_data):  # data为类AircraftDataset的一个实例,in_data同理
    input_dim = CFG.input_dim
    feature_fusion = np.zeros((len(in_data), 3, input_dim))
    features, coefs = feature_extration_from_his_unit_v1(his_data)  # shape(len(data),3,11)
    for in_unit in range(len(in_data)):
        data_q = in_data[in_unit]["input"].numpy()
        timeseries_q = in_data[in_unit]["timeseries"].numpy()
        feature_l = np.zeros((3, input_dim))
        for sensor in range(input_dim):
            w_unit_sensor = np.zeros(len(his_data))
            coef_q = np.polyfit(timeseries_q.reshape(-1), data_q[:, sensor], 2)
            piancha = data_q[:, sensor] - np.polyval(coef_q, timeseries_q.reshape(-1))
            sigma_j_2 = np.inner(piancha, piancha) / (len(timeseries_q) - 1)
            for his_unit in range(len(his_data)):
                residual_sum_ij = 0
                for time in range(len(timeseries_q)):
                    coef_ij = coefs[his_unit][sensor]
                    residual = (data_q[time][sensor] - np.polyval(coef_ij, time)) ** 2
                    residual_sum_ij += residual
                w_ij = np.exp(residual_sum_ij / (-2 * sigma_j_2))
                w_unit_sensor[his_unit] = w_ij
            feature_l[:, sensor] = features[:, :, sensor].T @ w_unit_sensor / sum(w_unit_sensor)
        feature_fusion[in_unit] = feature_l
    return feature_fusion


class AircraftFeature(Dataset):
    def __init__(self, feature):
        self.feature = feature

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        feature_l = self.feature[idx].transpose(1, 0).flatten()
        return torch.tensor(feature_l)
