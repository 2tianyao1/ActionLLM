import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import LinearNDInterpolator

# # 假设您已经有一个包含视频特征的二维NumPy数组，形状为 (old_length, feature_dim)
# features = np.load("/home/tianyao/tianyao_data/python_code/LLM_FUTR/data/features/rgb-01-1.npy")
# features = features.transpose()    #[11679,2048]
# print("features:",features.shape)
#
# # 定义新长度
# new_length = 11769
#
# # 创建插值函数
# interp_func = RectBivariateSpline(np.arange(features.shape[1]), np.arange(features.shape[0]), features.T, kx=1, ky=1)
#
# # 在新长度上进行插值
# x_new = np.linspace(0, features.shape[1] - 1, new_length)
# y_new = np.arange(features.shape[1])
# resized_features = interp_func(x_new, y_new)   #[11769]
#
# # 调整大小后的特征形状为 (new_length, feature_dim)
# print('resized_features:',resized_features.shape)

# 加载特征数据
features = np.load("/home/tianyao/tianyao_data/python_code/LLM_FUTR/data/features/rgb-01-1.npy")
features = features.transpose()  # [11679, 2048]

# 加载标签数据
with open("/home/tianyao/tianyao_data/python_code/LLM_FUTR/data/groundTruth/rgb-01-1.txt", "r") as f:
    labels = f.read().splitlines()

print("features shape:", features.shape)
print("labels length:", len(labels))

# 定义新长度
new_length = 11769

# 创建特征插值函数
interp_func = RectBivariateSpline(np.arange(features.shape[1]), np.arange(features.shape[0]), features.T, kx=1, ky=1)

# 在新长度上进行特征插值
x_new = np.linspace(0, features.shape[1] - 1, new_length)
y_new = np.arange(features.shape[1])
resized_features = interp_func(x_new, y_new)  # [11769, 2048]

# 插值后的标签列表
resized_labels = []

# 根据插值后的特征，按比例计算对应的标签值
for i in range(new_length):
    # 计算特征对应的原始索引
    old_index = int(round(x_new[i]))  #only get num from 0 to 2048,so resized_labels just have two class
    # 获取对应的标签
    label = labels[old_index]
    # 将标签添加到插值后的标签列表中
    resized_labels.append(label)

# 调整大小后的特征形状为 (new_length, feature_dim)，标签长度为 new_length
print('resized_features shape:', resized_features.shape)
print('resized_labels length:', len(resized_labels))