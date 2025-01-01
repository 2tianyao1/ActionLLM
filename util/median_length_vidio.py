import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

folder_path = '/home/ty/action_anticipation/LLM_FUTR/data/50_salads/groundTruth'  # 文件夹路径
# folder_path = '/home/tianyao/tianyao_data/python_code/LLM_FUTR/data/50_salads/groundTruth'
line_counts = []  # 存储非空行数的列表
n=0

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith('.txt'):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            non_empty_lines = [line for line in lines if line.strip()]
            line_count = len(non_empty_lines)
            line_counts.append(line_count)

# 计算行数的中位数
line_counts.sort()
n_lines = len(line_counts)
print(n_lines)
if n_lines % 2 == 0:
    median = (line_counts[n_lines // 2 - 1] + line_counts[n_lines // 2]) / 2
else:
    median = line_counts[n_lines // 2]

print("Line Counts:", line_counts)
print("Median:", median)

line_counts = np.array(line_counts)
bins = np.arange(0, 19000, 300)  # Set the width of the interval segment
hist, edges, patches = plt.hist(line_counts, bins=bins, alpha=0.7, color='blue', edgecolor='black')

x = edges[:-1] + (edges[1] - edges[0]) / 2  # 区间段的中心点
y = hist


# plt.xticks(np.arange(min(x), max(x) + 1, 600), rotation='vertical')
# plt.subplots_adjust(bottom=0.2)

plt.title('Histogram of video distribution for the 50s')
plt.xlabel('50s Video length')
plt.ylabel('Count')

fig = plt.gcf()
fig.tight_layout()

plt.show()

