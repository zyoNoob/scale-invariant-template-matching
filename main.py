import cv2
import os
import numpy as np # Import NumPy
from template_matching import ScaleInvariantSurfMatcher

def main():
    print("Running scale-invariant template matching example...")

    # Define paths relative to the script location
    script_dir = os.path.dirname(__file__)
    template_path = os.path.join(script_dir, "assets", "template.png")
    target_path = os.path.join(script_dir, "assets", "target.png")
    try:
        # 1. Initialize the matcher
        # You can adjust parameters like hessian_threshold if needed
        matcher = ScaleInvariantSurfMatcher(hessian_threshold=50)

        # 2. Perform matching using file paths
        print(f"\nMatching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}'")
        result_img, homography, corners, status = matcher.match(template_path, target_path, show_visualization=False)

        # 3. Process results
        print(f"Match Status: {status}")
        if status == "Detected":
            print("Template found!")
            print("Homography Matrix:\n", homography)
            print("Detected Corners (clipped):\n", corners.reshape(-1, 2))

            # Display the result
            cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection Result", 800, 600)
            # Draw the detected corners on the target image
            for corner in corners.reshape(-1, 2):
                cv2.circle(result_img, tuple(corner), 5, (0, 255, 0), -1)
            cv2.imshow("Detection Result", result_img)
            print("Displaying result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow("Detection Result")
        else:
            print(f"Template not found or matching failed: {status}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure template and target images exist.")
    except AttributeError as e:
         # This might happen if opencv-contrib-python is not installed correctly
         print(f"Error: {e}. Is 'opencv-contrib-python' installed and compatible?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed

if __name__ == "__main__":
    main()