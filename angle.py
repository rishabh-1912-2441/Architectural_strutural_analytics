import cv2
import numpy as np

# Load the image
image_path = '0025.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and find polygons
for contour in contours:
    # Approximate the contour to polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If the polygon has 4 corners (rectangle), calculate angles
    if len(approx) == 4:
        points = approx.reshape(-1, 2)
        
        # Calculate angles
        angles = []
        for i in range(4):
            pt1 = points[i]
            pt2 = points[(i + 1) % 4]
            pt3 = points[(i + 2) % 4]
            
            v1 = pt1 - pt2
            v2 = pt3 - pt2
            
            dot_product = np.dot(v1, v2)
            magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            angle = np.arccos(dot_product / magnitude_product)
            angle = np.degrees(angle)
            
            angles.append(angle)
        
        # Mark angles on the image
        for i, (pt, angle) in enumerate(zip(points, angles)):
            cv2.putText(image, f"{angle:.1f}°", tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw the contour
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Display the annotated image
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the annotated image
cv2.imwrite('annotated_image.jpg', image)
