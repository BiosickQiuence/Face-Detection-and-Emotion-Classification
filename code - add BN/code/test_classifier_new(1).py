import torch
import my_net

batch_size = 32

# 加载模型和数据
model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
PATH = 'code\\face_expression.pth'  # 确保路径正确
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型状态字典时允许缺失键
model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True), strict=False)
test_loader, classes = my_net.utility.loadTest("code\\images", batch_size)

model.eval()

# Step 1: Test in one batch
for X, y in test_loader:
    with torch.no_grad():
        # 运行模型并获得预测结果
        yHat = model(X)
        predicted_labels = torch.argmax(yHat, dim=1)

        # 显示前32个预测标签
        new_labels = predicted_labels.numpy()
        my_net.utility.imshow_with_labels(X[:batch_size], new_labels[:batch_size], classes)

        # 初始化统计变量
        TP = [0 for _ in range(len(classes))]
        FP = [0 for _ in range(len(classes))]
        FN = [0 for _ in range(len(classes))]
        total_correct_predictions = 0
        total_predictions = y.size(0)

        # 统计 TP, FP, FN
        for i in range(len(classes)):
            TP[i] += ((predicted_labels == i) & (y == i)).sum().item()
            FP[i] += ((predicted_labels == i) & (y != i)).sum().item()
            FN[i] += ((predicted_labels != i) & (y == i)).sum().item()

        # 计算总体准确率
        total_correct_predictions = sum(TP)
        total_accuracy = total_correct_predictions / total_predictions * 100
        print(f"Total accuracy for one batch: {total_accuracy:.2f}%")

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

            print(f"Class '{class_name}' for one batch: Accuracy = {accuracy_i:.2f}%, Recall = {recall_i:.2f}%")

    # 只处理一个批次，跳出循环
    break

def calculate_overall_recall(TP, FN):
    total_TP = sum(TP)
    total_FN = sum(FN)

    if total_TP + total_FN > 0:
        overall_recall = total_TP / (total_TP + total_FN) * 100
    else:
        overall_recall = 0

    return overall_recall

# Step 2: Evaluate on the entire test set
TP = [0 for _ in range(len(classes))]
FP = [0 for _ in range(len(classes))]
FN = [0 for _ in range(len(classes))]
total_correct_predictions = 0
total_predictions = 0

for X, y in test_loader:
    with torch.no_grad():
        yHat = model(X)
        predicted_labels = torch.argmax(yHat, dim=1)

        # 统计 TP, FP, FN
        for i in range(len(classes)):
            TP[i] += ((predicted_labels == i) & (y == i)).sum().item()
            FP[i] += ((predicted_labels == i) & (y != i)).sum().item()
            FN[i] += ((predicted_labels != i) & (y == i)).sum().item()

        total_correct_predictions += (predicted_labels == y).sum().item()
        total_predictions += y.size(0)

# 计算总体准确率
total_accuracy = total_correct_predictions / total_predictions * 100
print(f"Total accuracy on the entire test set: {total_accuracy:.2f}%")

# 计算总体召回率
overall_recall = calculate_overall_recall(TP, FN)
print(f"Overall recall on the entire test set: {overall_recall:.2f}%")

# 每个类别的准确率和召回率
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
