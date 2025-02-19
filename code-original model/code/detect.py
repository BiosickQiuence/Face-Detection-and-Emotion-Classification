import cv2
import my_net
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义图像转换
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

class Detector:
    def __init__(self, CascadePath, ModelPath):
        self.classes = ['happy', 'neutral', 'sad']  # 定义情感的列表
        # 加载级联分类器
        self.face_cascade = cv2.CascadeClassifier(CascadePath)  # 加载方法
        # 加载深度学习模型
        self.model, _, _ = my_net.classify.makeEmotionNet(False)  # 创建模型实例
        self.model.load_state_dict(torch.load(ModelPath, map_location=torch.device('cpu')))
        self.model.eval()  # 切换到评估模式

    def process(self, img):
        # 将图像转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # 存储裁剪的人脸图像
        cropped_images = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]  # 裁剪出人脸区域
            roi_gray = cv2.resize(roi_gray, (48, 48))  # 转化为48x48大小
            cropped_images.append(roi_gray)  # 添加到列表中
        
        # 转换所有裁剪图像为张量
        tensor_images = [self.transform2tensor(img) for img in cropped_images]
        
        # 预测每个张量并将结果绘制在图片上
        for i, tensor in enumerate(tensor_images):
            with torch.no_grad():
                output = self.model(tensor)
                _, predicted = torch.max(output.data, 1)  # 获取最大值的索引
                label = self.classes[predicted.item()]  # 根据索引获取预测的情感标签
                
                # 在图像上绘制情感标签
                cv2.putText(img, label, (faces[i][0], faces[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.rectangle(img, (faces[i][0], faces[i][1]), (faces[i][0]+faces[i][2], faces[i][1]+faces[i][3]), (255, 0, 0), 2)
        
        return img

    def transform2tensor(self, data):
        # 将灰度图像转换为三通道图像
        color_data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(color_data)  # 转换为PIL图像
        tensor_data = transform(pil_img)  # 应用预处理
        return tensor_data[None]  # 添加批处理维度

if __name__ == '__main__':
    # 读取演示图像
    input_image_path = 'demo.jpg'  # 测试用的图像文件
    img = cv2.imread(input_image_path)  # 读取图像
    if img is None:
        print(f"无法读取图像: {input_image_path}")
    else:
        # 初始化检测器
        detector = Detector('haarcascade_frontalface_default.xml', 'code\\face_expression.pth')
        
        # 处理图像
        result = detector.process(img)
        
        # 保存结果图像
        cv2.imwrite('result.jpg', result)
