import cv2
# Load pre-trained cascade file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read image and convert to grayscale
image = cv2.imread("C:\\Users\\30949\Desktop\\a.jpg")
# Convert to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(faces)
color=(255,0,0)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h),color)
cv2.imwrite('test1.png', image)


