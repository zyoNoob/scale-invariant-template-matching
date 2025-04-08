# Import necessary libraries
import cv2
import numpy as np
import os # To check if files exist

print(f"OpenCV version: {cv2.__version__}")

# --- Configuration ---
# Minimum number of good matches required to find the object
MIN_MATCH_COUNT = 10
# Lowe's ratio test threshold
LOWE_RATIO = 0.75
# Hessian Threshold for SURF detector
HESSIAN_THRESHOLD = 400
# Color for drawing the bounding box (BGR format)
DRAW_COLOR = (0, 255, 0) # Green

class ScaleInvariantSurfMatcher:
    """
    A class to perform scale-invariant template matching using SURF features
    and FlannBasedMatcher.

    Note: SURF is patented and might require a specific OpenCV build
    (opencv-contrib-python) and may have licensing restrictions for
    commercial use. This implementation assumes non-commercial, personal use.
    """

    def __init__(self, template_path):
        """
        Initializes the matcher with the template image.

        Args:
            template_path (str): The file path to the template image.

        Raises:
            FileNotFoundError: If the template image file does not exist.
            AttributeError: If the required SURF module is not found in the
                          current OpenCV installation (likely needs opencv-contrib-python).
            Exception: If the template image cannot be loaded or has no features.
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template image not found at: {template_path}")

        # Load the template image in grayscale
        self.template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template_img is None:
            raise Exception(f"Could not load template image from: {template_path}")

        # --- Initialize SURF Detector ---
        # Ensure SURF is available
        try:
            # Note: For older OpenCV versions, it might be cv2.SURF(HESSIAN_THRESHOLD)
            self.surf = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD)
        except AttributeError:
            print("\nError: cv2.xfeatures2d.SURF_create not found.")
            print("You might need to install 'opencv-contrib-python':")
            print("pip uninstall opencv-python")
            print("pip install opencv-contrib-python\n")
            raise AttributeError("SURF algorithm not available in this OpenCV build.")

        # Find keypoints and descriptors in the template image
        self.kp_template, self.desc_template = self.surf.detectAndCompute(self.template_img, None)

        if self.desc_template is None or len(self.kp_template) < MIN_MATCH_COUNT:
             raise Exception(f"Not enough features found in template image ({len(self.kp_template)} found). Try a lower Hessian threshold or a different template.")

        # --- Initialize FLANN Matcher ---
        # FLANN parameters for SURF (float descriptors)
        # Algorithm: KDTREE (index 0) - good for SIFT/SURF
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # 5 trees is a common value
        # Search parameters: specify how many times leafs should be checked
        search_params = dict(checks=50) # Higher checks = more accuracy, slower

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        print(f"Matcher initialized with template: {template_path} ({len(self.kp_template)} keypoints)")


    def match(self, target_image_path, draw_matches=False):
        """
        Finds the template in the target image.

        Args:
            target_image_path (str): The file path to the target image.
            draw_matches (bool): If True, draws the individual keypoint matches.
                                 (Can be slow and cluttered for many matches).

        Returns:
            tuple: A tuple containing:
                - result_image (numpy.ndarray): The target image (in color) with
                  a bounding box drawn around the detected template, or the
                  original color image if not found.
                - homography_matrix (numpy.ndarray or None): The 3x3 perspective
                  transformation matrix if found, otherwise None.
                - corners (numpy.ndarray or None): The coordinates of the
                  template corners in the target image if found, otherwise None.
                - status (str): A message indicating "Detected" or "Not Detected".

        Raises:
            FileNotFoundError: If the target image file does not exist.
            Exception: If the target image cannot be loaded or has no features.
        """
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found at: {target_image_path}")

        # Load the target image in color (for drawing) and grayscale (for detection)
        target_img_color = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
        target_img_gray = cv2.cvtColor(target_img_color, cv2.COLOR_BGR2GRAY)
        if target_img_color is None:
             raise Exception(f"Could not load target image from: {target_image_path}")

        # Find keypoints and descriptors in the target image
        kp_target, desc_target = self.surf.detectAndCompute(target_img_gray, None)

        if desc_target is None or len(kp_target) < MIN_MATCH_COUNT:
            print(f"Warning: Not enough features found in target image ({len(kp_target)} found).")
            return target_img_color, None, None, "Not Detected (Insufficient target features)"

        # --- Match descriptors using FLANN ---
        # k=2 means find the two best matches for each descriptor
        matches = self.flann.knnMatch(self.desc_template, desc_target, k=2)

        # --- Filter matches using Lowe's ratio test ---
        good_matches = []
        for m, n in matches:
            # Ensure the distance ratio is below the threshold
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)

        print(f"Found {len(kp_target)} keypoints in target.")
        print(f"Found {len(matches)} raw matches.")
        print(f"Found {len(good_matches)} good matches after ratio test.")

        # --- Find Homography if enough good matches exist ---
        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find the perspective transformation matrix
            # RANSAC helps filter out outliers
            homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 5.0 is the RANSAC reprojection threshold (max distance to be considered an inlier)

            if homography_matrix is not None:
                # Apply perspective transformation to the template corners
                h, w = self.template_img.shape[:2] # Get height and width
                template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, homography_matrix)

                # Draw the bounding box around the detected object in the target image
                # Ensure corners are integers for drawing
                target_img_color = cv2.polylines(target_img_color, [np.int32(transformed_corners)], True, DRAW_COLOR, 3, cv2.LINE_AA)

                status = "Detected"
                print(f"Object detected. Homography found. Inliers: {np.sum(mask)}/{len(good_matches)}")

                # Optional: Draw the matches themselves
                if draw_matches:
                   # Create an output image showing the matches
                   img_matches = cv2.drawMatches(self.template_img, self.kp_template, target_img_gray, kp_target, good_matches, None,
                                                  matchColor=(0, 255, 0), # draw matches in green color
                                                  singlePointColor=None,
                                                  matchesMask=mask.ravel().tolist(), # draw only inliers
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                   # You might want to display or save img_matches separately
                   # cv2.imshow("Matches", img_matches)
                   # cv2.waitKey(0)
                   # cv2.destroyWindow("Matches")
                   pass # Placeholder - decide how to handle the match image


                return target_img_color, homography_matrix, np.int32(transformed_corners), status
            else:
                print("Warning: Homography could not be computed.")
                status = "Not Detected (Homography failed)"
                return target_img_color, None, None, status
        else:
            print(f"Warning: Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
            status = "Not Detected (Insufficient good matches)"
            return target_img_color, None, None, status


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy image files for demonstration if they don't exist
    # In a real scenario, replace these paths with your actual image paths
    template_file = "template.png"
    target_file = "target_with_template.png"

    # Create a simple template (e.g., a gray square with a white circle)
    if not os.path.exists(template_file):
        template = np.zeros((100, 100), dtype=np.uint8) + 50 # Gray background
        cv2.circle(template, (50, 50), 30, (200), -1) # White-ish circle
        cv2.imwrite(template_file, template)
        print(f"Created dummy template: {template_file}")

    # Create a larger target image containing the template scaled and rotated
    if not os.path.exists(target_file):
        target = np.zeros((400, 500), dtype=np.uint8) + 100 # Different gray background
        # Define scale and rotation
        scale = 1.5
        angle = 30
        h, w = template.shape[:2]
        # Get rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Adjust translation part of the matrix to place it somewhere
        M[0, 2] += (250 - center[0] * scale - M[0,1]*center[1]) # Rough centering adjustment
        M[1, 2] += (200 - center[1] * scale + M[0,0]*center[1])
        # Apply affine transformation
        transformed_template = cv2.warpAffine(template, M, (int(w*scale*1.5), int(h*scale*1.5))) # Make output size larger just in case
        # Place the transformed template onto the target - simple pasting
        th, tw = transformed_template.shape[:2]
        roi_x, roi_y = 150, 100 # Top-left corner to paste
        # Ensure ROI is within bounds
        roi_h = min(th, target.shape[0] - roi_y)
        roi_w = min(tw, target.shape[1] - roi_x)
        # Simple pasting (ignoring transparency/masking for this example)
        target[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = transformed_template[:roi_h, :roi_w]

        cv2.imwrite(target_file, target)
        print(f"Created dummy target image: {target_file}")

    try:
        # 1. Initialize the matcher with the template
        matcher = ScaleInvariantSurfMatcher(template_path=template_file)

        # 2. Perform matching on the target image
        result_img, H, corners, status = matcher.match(target_image_path=target_file, draw_matches=False)

        # 3. Display the result
        print(f"\nMatching Status: {status}")
        if corners is not None:
            print("Detected Corners:")
            print(corners.reshape(-1, 2)) # Reshape for easier reading
            # print("\nHomography Matrix:")
            # print(H)

        # Display the image with the bounding box (if found)
        cv2.imshow("Matching Result", result_img)
        print("\nPress any key to close the image window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except AttributeError as e:
        print(f"Error: {e}. Ensure opencv-contrib-python is installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

