import cv2
import numpy as np
import os

def create_silhouette(width, height, filename, outline_color=(255, 255, 255), bg_color=(30, 30, 30)):
    """Create a simple silhouette image for posture visualization."""
    # Create a blank image with dark background
    image = np.ones((height, width, 3), dtype=np.uint8)
    image[:] = bg_color  # Set background color
    
    # Draw silhouette based on the view
    if "front" in filename or "back" in filename:
        # Front/Back view - draw a simple human silhouette
        # Head
        cv2.circle(image, (width//2, height//6), height//12, outline_color, 2)
        
        # Neck to hips
        cv2.line(image, (width//2, height//6 + height//12), (width//2, height//2), outline_color, 2)
        
        # Shoulders
        cv2.line(image, (width//2 - width//6, height//3), (width//2 + width//6, height//3), outline_color, 2)
        
        # Arms
        cv2.line(image, (width//2 - width//6, height//3), (width//2 - width//4, height//2), outline_color, 2)
        cv2.line(image, (width//2 + width//6, height//3), (width//2 + width//4, height//2), outline_color, 2)
        
        # Legs
        cv2.line(image, (width//2, height//2), (width//2 - width//8, height - height//6), outline_color, 2)
        cv2.line(image, (width//2, height//2), (width//2 + width//8, height - height//6), outline_color, 2)
        
    else:  # side view (left/right)
        # Side view silhouette
        # Head
        cv2.circle(image, (width//2, height//6), height//12, outline_color, 2)
        
        # Neck to back
        cv2.line(image, (width//2, height//6 + height//12), (width//2, height//3), outline_color, 2)
        
        # Spine
        cv2.line(image, (width//2, height//3), (width//2 - width//20, height//2), outline_color, 2)
        
        # Arm
        cv2.line(image, (width//2, height//3), (width//2 + width//6, height//3), outline_color, 2)
        cv2.line(image, (width//2 + width//6, height//3), (width//2 + width//6, height//2), outline_color, 2)
        
        # Leg
        cv2.line(image, (width//2 - width//20, height//2), (width//2 - width//15, height - height//6), outline_color, 2)
    
    # Add text
    if "front" in filename:
        cv2.putText(image, "Front View", (width//4, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, outline_color, 2)
    elif "back" in filename:
        cv2.putText(image, "Back View", (width//4, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, outline_color, 2)
    elif "left" in filename:
        cv2.putText(image, "Left View", (width//4, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, outline_color, 2)
    elif "right" in filename:
        cv2.putText(image, "Right View", (width//4, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, outline_color, 2)
    
    # Save the image
    cv2.imwrite(filename, image)
    print(f"Created {filename}")

def main():
    # Create directory if it doesn't exist
    img_dir = os.path.join("flaskProject1", "app", "static", "img")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create four silhouette images
    width, height = 300, 500
    create_silhouette(width, height, os.path.join(img_dir, "front_silhouette.png"))
    create_silhouette(width, height, os.path.join(img_dir, "back_silhouette.png"))
    create_silhouette(width, height, os.path.join(img_dir, "left_silhouette.png"))
    create_silhouette(width, height, os.path.join(img_dir, "right_silhouette.png"))

if __name__ == "__main__":
    main() 