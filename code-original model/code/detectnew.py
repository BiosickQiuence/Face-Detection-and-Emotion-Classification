import cv2
import my_net
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

# Define transformation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

class Detector:
    def __init__(self, CascadePath, ModelPath):
        self.classes = ['happy', 'neutral', 'sad']
        # Load a CascadeClassifier for face detection
        cascade_path = 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # Load a pre-trained deep learning model
        ModelPath = 'code\\face_expression.pth'
        self.model, _, _ = my_net.classify.makeEmotionNet(False)
        self.model.load_state_dict(torch.load(ModelPath, map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

    def process(self, img):
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
        print(f"Number of faces detected: {len(faces)}")
        for (x, y, w, h) in faces:
            # Crop the detected face and resize it to 48x48
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Convert the face to tensor
            tensor_face = self.transform2tensor(face_resized)
            
            # Predict the emotion using the model
            with torch.no_grad():
                output = self.model(tensor_face)
                predicted_label = torch.argmax(output, dim=1).item()
                emotion = self.classes[predicted_label]
            
            # Draw the prediction result on the image
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
            
        return img

    def transform2tensor(self, data):
        # Convert the image to PIL format and apply transformations
        pil_img = Image.fromarray(data)
        tensor_data = transform(pil_img)
        return tensor_data[None]  # Add batch dimension

if __name__ == '__main__':
    # Path to the cascade file and model
    cascade_path = 'haarcascade_frontalface_default.xml'
    model_path = 'code\\face_expression.pth'
    
    # Read the demo image
    input_image_path = 'demo.jpg'
    output_image_path = 'output_image.jpg'
    img = cv2.imread(input_image_path)
    
    # Initialize the detector
    detector = Detector(cascade_path, model_path)
    
    # Process the image
    result_img = detector.process(img)
    
    # Save the result image
    cv2.imwrite(output_image_path, result_img)
    