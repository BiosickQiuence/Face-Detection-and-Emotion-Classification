import torch
import my_net
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

batch_size = 32

# 加载模型和数据
model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
PATH = 'code\\face_expression.pth'
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
test_loader, classes = my_net.utility.loadTest("code\\images", batch_size)

model.eval()

# 初始化列表以记录真实标签和预测标签
all_true_labels = []
all_predicted_labels = []

# 遍历整个测试集
for X, y in test_loader:
    with torch.no_grad():
        # 运行模型并获得预测结果
        yHat = model(X)
        # 获取预测标签
        predicted_labels = torch.argmax(yHat, dim=1)

        # 记录真实标签和预测标签
        all_true_labels.extend(y.numpy())
        all_predicted_labels.extend(predicted_labels.numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_true_labels, all_predicted_labels)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# 接下来的性能评估代码保持不变
TP = [0 for _ in range(len(classes))]
FP = [0 for _ in range(len(classes))]
FN = [0 for _ in range(len(classes))]
total_correct_predictions = 0
total_predictions = 0

# 统计 TP, FP, FN
for i in range(len(classes)):
    TP[i] += ((np.array(all_predicted_labels) == i) & (np.array(all_true_labels) == i)).sum()
    FP[i] += ((np.array(all_predicted_labels) == i) & (np.array(all_true_labels) != i)).sum()
    FN[i] += ((np.array(all_predicted_labels) != i) & (np.array(all_true_labels) == i)).sum()

# 计算总体准确率
total_correct_predictions = sum(TP)
total_predictions = len(all_true_labels)

total_accuracy = total_correct_predictions / total_predictions * 100
print(f"Total accuracy on the entire test set: {total_accuracy:.2f}%")

# 计算每个类别的准确率和召回率
for i, class_name in enumerate(classes):
    if TP[i] + FP[i] > 0:
        accuracy_i = TP[i] / (TP[i] + FP[i]) * 100
    else:
        accuracy_i = 0

    if TP[i] + FN[i] > 0:
        recall_i = TP[i] / (TP[i] + FN[i]) * 100
    else:
        recall_i = 0

    print(f"Class '{class_name}' on the entire test set: Accuracy = {accuracy_i:.2f}%, Recall = {recall_i:.2f}%")
