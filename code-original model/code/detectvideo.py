import cv2
import os
import torch
from single_model import Detector

class Detector:
    def __init__(self, cascade_path, model_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

        self.model = self.load_model(model_path)  # 加载模型

    def load_model(self, model_path):
        return torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图像
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)  # 检测人脸
        return [{'box': (x, y, w, h), 'emotion': 'Unknown'} for (x, y, w, h) in faces]  # 返回人脸及情感信息


def process_video(input_video_path, output_video_path):
    cascade_path = 'haarcascade_frontalface_default.xml'  # 级联分类器路径
    model_path = 'code\\face_expression.pth'  # 模型文件路径
    
    # 检查模型文件是否存在
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    detector = Detector(cascade_path, model_path)  # 实例化 Detector
    cap = cv2.VideoCapture(input_video_path)  # 打开视频文件

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象，使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人脸并获取框和情感
        faces = detector.detect_faces(frame)

        # 绘制检测到的人脸框和情感标签
        for face in faces:
            x, y, w, h = face['box']  # 获取人脸框坐标
            emotion = face.get('emotion', 'Unknown')  # 使用 get 方法避免 KeyError
            
            # 绘制矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 在框上方写入情感标签
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # 将处理后的视频帧写入文件
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video('model.mp4', 'output_video.mp4v')  # 输入和输出文件路径
