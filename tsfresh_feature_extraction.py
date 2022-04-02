from utilities import *
from config import *
from data_preparing import *
from DNN_model import *
#import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters,EfficientFCParameters
from tsfresh import feature_extraction
from tsfresh import extract_features
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA
import tsfresh
from tsfresh import feature_selection
from tsfresh import extract_features, select_features

seed_everything(CFG.seed)
DATA_PATH = "./Data_FD003/preprocessed data/"
attribute = ['Unit', 'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30',
             'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
df_train = pd.read_csv(DATA_PATH + 'TD_data.csv', names=attribute, header=None)
df_test = pd.read_csv(DATA_PATH + 'Test_data.csv', names=attribute, header=None)

df_train = Preprocessing.add_timeseries(df_train)
df_test = Preprocessing.add_timeseries(df_test)

train_label=pd.read_csv(DATA_PATH +"TD_mode.csv", header=None).values
test_label=pd.read_csv(DATA_PATH +"Test_mode.csv", header=None).values

train_dataset=AircraftDataset(df_train,train_label) # 不插0计算创建dataset的子类
test_dataset = AircraftDataset(df_test,test_label)

train_dataset_expend=AircraftDataset_expend(df_train,train_label,False) # 构建切割后的训练集

# #提取feature map
# kind_to_fc_parameters=load_dict("./numpy/kind_to_fc_parameters")
# dataframe_list=[]
# for idx in range(len(train_dataset_expend)):
#     dataframe_list.append(get_dataframe_ofcuttedsequence(train_dataset_expend,idx))
# train_dataset_expend_dataframe=pd.concat(dataframe_list).reset_index(drop=True)
# train_dataset_expend_dataframe.to_csv("./numpy/train_dataset_expend_dataframe.csv")
# extracted_features_enhanced = extract_features(train_dataset_expend_dataframe,
#                                               column_id="Unit",
#                                               column_sort="Time",
#                                              kind_to_fc_parameters =kind_to_fc_parameters)
# extracted_features_enhanced=impute(extracted_features_enhanced)
#
# #PCA降维
# pca = PCA(n_components=30,whiten=True)
# extracted_features_PCA=pca.fit_transform(extracted_features_enhanced.values)
# np.save("./numpy/extracted_features_PCA.npy",extracted_features_PCA)



if __name__ == "__main__":
  #提取feature map
  kind_to_fc_parameters=load_dict("./numpy/kind_to_fc_parameters_new")
  dataframe_list=[]
  for idx in range(len(train_dataset_expend)):
      dataframe_list.append(get_dataframe_ofcuttedsequence(train_dataset_expend,idx))
  train_dataset_expend_dataframe=pd.concat(dataframe_list).reset_index(drop=True)
  train_dataset_expend_dataframe.to_csv("./numpy/train_dataset_expend_dataframe.csv")
  # train_dataset_expend_dataframe = pd.read_csv("./numpy/train_dataset_expend_dataframe.csv")
  # train_dataset_expend_dataframe = train_dataset_expend_dataframe.drop(["Unnamed: 0"], axis=1)
  extracted_features_enhanced = extract_features(train_dataset_expend_dataframe,
                                              column_id="Unit",
                                              column_sort="Time",
                                             kind_to_fc_parameters =kind_to_fc_parameters)
  extracted_features_enhanced=impute(extracted_features_enhanced)

  #PCA降维
  pca = PCA(n_components=80,whiten=True)
  extracted_features_PCA=pca.fit_transform(extracted_features_enhanced.values)
  np.save("./numpy/extracted_features_PCA.npy",extracted_features_PCA)

  #test data
  dataframe_test_list = []
  for idx in range(len(test_dataset)):
      dataframe_test_list.append(get_dataframe_ofcuttedsequence(test_dataset, idx))
  test_dataset_dataframe = pd.concat(dataframe_test_list).reset_index(drop=True)
  test_dataset_dataframe.to_csv("./numpy/test_dataset_dataframe.csv")
  # test_dataset_dataframe = pd.read_csv("./numpy/test_dataset_dataframe.csv")
  # test_dataset_dataframe = test_dataset_dataframe.drop(["Unnamed: 0"], axis=1)
  test_extracted_features = extract_features(test_dataset_dataframe,
                                             column_id="Unit",
                                             column_sort="Time",
                                             kind_to_fc_parameters=kind_to_fc_parameters)
  test_extracted_features = impute(test_extracted_features)
  test_feature_PCA=pca.transform(test_extracted_features.values)
  np.save("./numpy/test_feature_PCA.npy", test_feature_PCA)





