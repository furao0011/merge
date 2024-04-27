import numpy as np
from sklearn.model_selection import train_test_split

# 假设您有1807张图像，文件名为从1到1807的数字
image_filenames = np.arange(1, 1808)

# 将图像文件名分割为训练集和验证集，这里使用80%的数据作为训练集，20%的数据作为验证集
X_train, X_val = train_test_split(image_filenames, test_size=0.2, random_state=20, shuffle=True)

# 将训练集文件名写入train.txt
with open('train.txt', 'w') as file:
    for filename in X_train:
        file.write(str(filename) + '\n')  # 将filename转换为字符串

# 将验证集文件名写入val.txt
with open('val.txt', 'w') as file:
    for filename in X_val:
        file.write(str(filename) + '\n')  # 将filename转换为字符串