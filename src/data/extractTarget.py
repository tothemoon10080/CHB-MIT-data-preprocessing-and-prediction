import os
def extractTarget(summary_file_path, edf_file_path):
    edf_file_name = os.path.basename(edf_file_path)
    # 初始化变量
    seizure_start_time = None
    seizure_end_time = None
    # 打开并读取文本文件
    with open(summary_file_path, 'r') as file:
        lines = file.readlines()
    # 标记是否找到匹配的文件名
    found = False
    # 遍历文件中的每一行
    for line in lines:
        if "File Name: " + edf_file_name in line:
            found = True
        if found:
            if "Number of Seizures in File: 0" in line:
                return None, None  # 没有癫痫发作，直接返回 None
            if "Seizure Start Time:" in line:
                seizure_start_time = int(line.split(": ")[1].split(" ")[0])
            if "Seizure End Time:" in line:
                seizure_end_time = int(line.split(": ")[1].split(" ")[0])
                break  # 找到所需的信息后退出循环
    return seizure_start_time, seizure_end_time

