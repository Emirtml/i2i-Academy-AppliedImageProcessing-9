import cv2
import numpy as np
import easyocr

# 1. Load the car image from the folder
# Make sure your image file name is exactly 'araba.jpg'
image_path = 'araba.jpg'
original_image = cv2.imread(image_path)

# Convert the image to grayscale (black and white)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# 2. Preprocessing (Blur to reduce noise)
# Gaussian Blur helps to smooth out small dirt and background text
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged_image = cv2.Canny(blurred_image, 50, 150)

# 3. Find all contours (shapes) from the edged image
contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort contours by size from biggest to smallest
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

plate_contour = None
x, y, w, h = 0, 0, 0, 0

# 4. Loop through shapes to find the rectangle plate
for c in contours:
    perimeter = cv2.arcLength(c, True)
    # 0.04 factor simplifies the shape edges to detect 4 corners easily
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    
    # Check if the simplified shape has exactly 4 corners (Rectangle)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        
        # Check if the shape matches standard horizontal license plate ratios
        if 3.0 <= aspect_ratio <= 5.5:
            plate_contour = approx
            break

# 5. Crop the plate and read text using Artificial Intelligence (EasyOCR)
if plate_contour is not None:
    # Draw a green box around the plate on the original image
    cv2.drawContours(original_image, [plate_contour], -1, (0, 255, 0), 3)
    
    # Crop operation (Cutting out the plate area)
    cropped_plate = original_image[y:y+h, x:x+w]
    
    print("\n--- AI OCR Reading Process Started ---")
    # Initialize EasyOCR for English characters
    reader = easyocr.Reader(['en'], gpu=False)
    # Read text from the cropped image
    ocr_result = reader.readtext(cropped_plate)
    
    # If text is successfully found, print it to terminal
    if ocr_result:
        plate_text = ocr_result[0][1]
        print(f"SUCCESS: Recognized License Plate String -> {plate_text}")
    else:
        print("OCR could not read text from the cropped image.")
    print("---------------------------------------\n")
    
    # 6. Show visualization windows on screen
    cv2.imshow('1 - Preprocessed Edges (Canny)', edged_image)
    cv2.imshow('2 - Detected License Plate Box', original_image)
    cv2.imshow('3 - Cropped Plate Area Only', cropped_plate)
else:
    print("Error: License plate contour could not be detected. Adjust image or parameters.")

# Wait for any key press to close windows cleanly
cv2.waitKey(0)
cv2.destroyAllWindows()