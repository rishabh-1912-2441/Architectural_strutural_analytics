import os
import cv2

# Function to classify shapes into groups based on Hu Moments
def classify_shapes(input_folder):
    # Dictionary to store shape groups
    shape_groups = {}

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Construct the full path to the input image
            image_path = os.path.join(input_folder, filename)
            
            try:
                # Load the input image
                image = cv2.imread(image_path)
                
                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Threshold the image to get binary image
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                
                # Find contours in the binary image
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate Hu Moments for the contour
                moments = cv2.moments(contours[0])
                hu_moments = cv2.HuMoments(moments)
                
                # Flatten Hu Moments to create a feature vector
                feature_vector = tuple(hu_moments.flatten())
                
                # Check if the feature vector already exists in the dictionary
                if feature_vector in shape_groups:
                    # If the group already exists, append the image to its folder
                    group_folder = shape_groups[feature_vector]
                else:
                    # If the group doesn't exist, create a new folder for the group
                    group_folder = os.path.join(input_folder, f"group_{len(shape_groups) + 1}")
                    os.makedirs(group_folder)
                    shape_groups[feature_vector] = group_folder
                
                # Save the image into the corresponding group folder
                output_image_path = os.path.join(group_folder, filename)
                cv2.imwrite(output_image_path, image)
                
                print(f"Image {filename} saved in group {feature_vector}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return shape_groups

# Function to select one representative image from each group and save them into a single folder
def save_representative_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Classify shapes into groups
    shape_groups = classify_shapes(input_folder)
    
    # Counter for naming the output images
    image_counter = 1
    
    # Loop through the shape groups
    for feature_vector, group_folder in shape_groups.items():
        # Get a list of images in the group folder
        group_images = os.listdir(group_folder)
        
        # Select the first image in the group folder as the representative image
        representative_image = group_images[0]
        
        # Construct the path to the representative image
        representative_image_path = os.path.join(group_folder, representative_image)
        
        # Read the representative image
        image = cv2.imread(representative_image_path)
        
        # Save the representative image to the output folder
        output_image_path = os.path.join(output_folder, f"{image_counter}.png")
        cv2.imwrite(output_image_path, image)
        
        # Increment the image counter
        image_counter += 1

# Example usage
input_folder = 'b'
output_folder = 'h'

# Save one representative image from each group into a single folder
save_representative_images(input_folder, output_folder)
