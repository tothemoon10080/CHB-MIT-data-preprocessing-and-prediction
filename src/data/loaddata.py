import glob
import os.path
import numpy as np
from src.data.extractFeture import preprocess_and_extract_features_mne_with_timestamps
from src.data.extractTarget import extractTarget

def extract_data_and_labels(edf_file_path, summary_file_path):

    # 提取特征
    X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path)
    # 提取标签
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    y = np.array([1 if seizure_start_time <= row[0] <= seizure_end_time else 0 for row in X])

    #从X数组中移除第一列Time
    X = X[:,1:]
    return X,y


def load_data(subject_id,base_path):
    """
    加载给定主题的数据。
    会读取给定chb主题的所有edf文件，并从每个文件中提取特征。
    返回一个包含所有数据的列表，以及一个包含所有标签的列表。
    其中，每个数据都是一个形状为 (n_samples, n_features) 的数组，每个标签都是一个形状为 (n_samples,) 的数组。
    """
    edf_file_path = sorted(glob.glob(os.path.join(base_path, "chb{:02d}/*.edf".format(subject_id))))
    summary_file_path = os.path.join(base_path, "chb{:02d}/chb{:02d}-summary.txt".format(subject_id, subject_id))
    all_X = []
    all_y = []
    for edf_file_path in edf_file_path:
        X, y = extract_data_and_labels(edf_file_path, summary_file_path)
        all_X.append(X)
        all_y.append(y)
    return all_X,all_y

#使用方法：
# subject_id = 1
# base_path = "data"
# all_X,all_y = load_data(subject_id,base_path)

#对于all_y每个数据，统计1和0的个数并打印
# total_n_count = 0
# total_p_count = 0
# for y in all_y:
#     p_count = 0
#     n_count = 0
#     for lable in y:
#         if lable == 1:
#             p_count += 1
#         else:
#             n_count += 1
#     total_n_count += n_count
#     total_p_count += p_count
# print("total_p_count/total_count:",total_p_count/(total_n_count+total_p_count))

## total_p_count/total_count: 0.018808777429467086