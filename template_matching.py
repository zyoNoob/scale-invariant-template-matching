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
# Hessian Threshold for SURF detector - User reported changing this to 100
HESSIAN_THRESHOLD = 100 # Adjusted based on user feedback
# Color for drawing the bounding box (BGR format)
DRAW_COLOR = (0, 255, 0) # Green
# Color for drawing good matches
MATCH_COLOR = (0, 255, 0) # Green

class ScaleInvariantSurfMatcher:
    """
    A class to perform scale-invariant template matching using SURF features
    and FlannBasedMatcher, including visualization of good matches and
    ensuring the drawn bounding box stays within image bounds.

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
        try:
            # Check if xfeatures2d is available
            if not hasattr(cv2, 'xfeatures2d'):
                 raise AttributeError("cv2.xfeatures2d module not found. Install 'opencv-contrib-python'.")
            self.surf = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD)
            print(f"Using SURF with Hessian Threshold: {HESSIAN_THRESHOLD}")
        except AttributeError as e:
            print(f"\nError initializing SURF: {e}")
            print("You might need to install 'opencv-contrib-python' (pip install opencv-contrib-python)")
            print("OR rebuild OpenCV with non-free modules enabled.")
            print("Ensure you have the correct version compatible with your base OpenCV.")
            raise AttributeError("SURF algorithm not available or failed to initialize.")

        # Find keypoints and descriptors in the template image
        self.kp_template, self.desc_template = self.surf.detectAndCompute(self.template_img, None)

        if self.desc_template is None or len(self.kp_template) < 2: # Need at least 2 for knnMatch k=2
            raise Exception(f"Not enough features found in template image ({len(self.kp_template)} found). Try a lower Hessian threshold or a different template.")

        # --- Initialize FLANN Matcher ---
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50) # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        print(f"Matcher initialized with template: {template_path} ({len(self.kp_template)} keypoints)")


    def match(self, target_image_path):
        """
        Finds the template in the target image, draws a bounding box (clipped
        to image bounds), and visualizes matches.

        Args:
            target_image_path (str): The file path to the target image.

        Returns:
            tuple: A tuple containing:
                - result_image (numpy.ndarray): The target image (in color) with
                  a bounding box drawn around the detected template (clipped to
                  image bounds), or the original color image if not found.
                - homography_matrix (numpy.ndarray or None): The 3x3 perspective
                  transformation matrix if found, otherwise None.
                - clipped_corners (numpy.ndarray or None): The coordinates of the
                  template corners in the target image (clipped to image bounds)
                  if found, otherwise None.
                - status (str): A message indicating "Detected" or "Not Detected".

        Raises:
            FileNotFoundError: If the target image file does not exist.
            Exception: If the target image cannot be loaded or has no features.
        """
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found at: {target_image_path}")

        # Load the target image in color (for drawing) and grayscale (for detection)
        target_img_color = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
        if target_img_color is None:
             raise Exception(f"Could not load target image from: {target_image_path}")
        target_img_gray = cv2.cvtColor(target_img_color, cv2.COLOR_BGR2GRAY)
        target_h, target_w = target_img_color.shape[:2] # Get target dimensions for clipping

        # Find keypoints and descriptors in the target image
        kp_target, desc_target = self.surf.detectAndCompute(target_img_gray, None)

        if desc_target is None or len(kp_target) < 2: # Need at least 2 for knnMatch k=2
            print(f"Warning: Not enough features found in target image ({len(kp_target)} found).")
            return target_img_color, None, None, "Not Detected (Insufficient target features)"

        # --- Match descriptors using FLANN ---
        # Ensure descriptors are float32, as required by FLANN
        if self.desc_template.dtype != np.float32:
            self.desc_template = np.float32(self.desc_template)
        if desc_target.dtype != np.float32:
            desc_target = np.float32(desc_target)

        matches = self.flann.knnMatch(self.desc_template, desc_target, k=2)

        # Filter matches using Lowe's ratio test
        # Store all matches that pass the ratio test, handling cases where k=1 might occur
        good_matches = []
        for i, m_n in enumerate(matches):
            # Check if m_n is not empty and contains at least two matches
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)
            elif len(m_n) == 1:
                # Handle cases where only one match is found (less common)
                # Optionally, consider adding m_n[0] if you want to be less strict
                # good_matches.append(m_n[0])
                pass # Exclude single matches for robustness

        print(f"Found {len(kp_target)} keypoints in target.")
        print(f"Found {len(matches)} raw potential match pairs.")
        print(f"Found {len(good_matches)} good matches after ratio test.")

        # --- Visualize the good matches BEFORE attempting homography ---
        try:
            # Draw only the good matches found by the ratio test
            img_matches_visualization = cv2.drawMatches(
                self.template_img, self.kp_template,
                target_img_gray, kp_target, # Use grayscale target for visualization consistency
                good_matches, None, # Pass the filtered list
                matchColor=MATCH_COLOR,
                singlePointColor=None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # Don't draw lone keypoints
            )

            # Display the visualization window
            cv2.imshow("Good Matches Visualization", img_matches_visualization)
            print("\n--- VISUALIZATION ---")
            print("Showing the 'good' matches (passed ratio test). Inspect the lines.")
            print("Press any key in the 'Good Matches Visualization' window to continue...")
            cv2.waitKey(0) # Wait indefinitely until a key is pressed
            cv2.destroyWindow("Good Matches Visualization") # Close the window
            print("--- Continuing with Homography ---")
        except cv2.error as e:
            print(f"Error during match visualization: {e}")
            print("This might happen with certain OpenCV versions or if keypoints/matches are invalid.")


        # --- Find Homography if enough good matches exist ---
        homography_matrix = None
        clipped_corners = None
        status = "Not Detected"

        if len(good_matches) >= MIN_MATCH_COUNT:
            # Extract location of good matches
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find the perspective transformation matrix using RANSAC
            homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 5.0 is the RANSAC reprojection threshold

            if homography_matrix is not None:
                # Apply perspective transformation to the template corners
                h, w = self.template_img.shape[:2]
                template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, homography_matrix)

                # --- Clip corners to target image boundaries ---
                # Ensure transformed_corners is not None before clipping
                if transformed_corners is not None:
                    # Reshape for easier iteration and convert to integer temporarily for clipping
                    corners_int = np.int32(transformed_corners.reshape(-1, 2))

                    # Clip each coordinate (x, y)
                    clipped_corners_list = []
                    for x, y in corners_int:
                        clipped_x = max(0, min(x, target_w - 1)) # Clip x between 0 and width-1
                        clipped_y = max(0, min(y, target_h - 1)) # Clip y between 0 and height-1
                        clipped_corners_list.append([clipped_x, clipped_y])

                    # Convert back to the required shape (N, 1, 2) and type (int32 for polylines)
                    clipped_corners = np.array(clipped_corners_list, dtype=np.int32).reshape(-1, 1, 2)

                    # Draw the BOUNDED polygon on the color target image
                    target_img_color = cv2.polylines(target_img_color, [clipped_corners], True, DRAW_COLOR, 3, cv2.LINE_AA)

                    status = "Detected"
                    # Count inliers based on the mask returned by findHomography
                    inlier_count = np.sum(mask) if mask is not None else 0
                    print(f"Object detected. Homography found. RANSAC Inliers: {inlier_count}/{len(good_matches)}")
                else:
                     print("Warning: perspectiveTransform returned None. Cannot draw bounding box.")
                     status = "Not Detected (Transform failed)"


            else:
                print("Warning: Homography could not be computed (findHomography returned None).")
                status = "Not Detected (Homography failed)"
                homography_matrix = None # Ensure it's None if failed
                clipped_corners = None
        else:
            print(f"Warning: Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
            status = "Not Detected (Insufficient good matches)"
            homography_matrix = None # Ensure it's None if failed
            clipped_corners = None

        return target_img_color, homography_matrix, clipped_corners, status