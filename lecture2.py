import cv2
image = cv2.imread('D:\\pythonProject\\test.png')
top_left = (80, 60)
bottom_right = (80 + 480, 60 + 360)
color=(255,0,0)
cv2.rectangle(image, top_left, bottom_right,color)
cropped_image = image[30:450, 40:600]
original_height, original_width = image.shape[:2]
new_width = original_width // 2
new_height = original_height // 2
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
text="I love SI100B"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = (original_width - text_size[0]) // 2
text_y = (original_height + text_size[1]) // 2
color = (255, 255, 255)
cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
cv2.imwrite("result.png", image)