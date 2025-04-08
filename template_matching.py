# Import necessary libraries
import cv2
import numpy as np
import os # To check if files exist

# --- Configuration ---
# Minimum number of good matches required to find the object
MIN_MATCH_COUNT = 5
# Lowe's ratio test threshold
LOWE_RATIO = 1.0
# Hessian Threshold for SURF detector - User reported changing this to 100
HESSIAN_THRESHOLD = 50 # Adjusted based on user feedback
# Color for drawing the bounding box (BGR format)
DRAW_COLOR = (0, 255, 0) # Green
# Color for drawing good matches
MATCH_COLOR = (0, 255, 0) # Green
# Flag to control match visualization popup window
SHOW_MATCH_VISUALIZATION = False # Set to True to show the match visualization window

class ScaleInvariantSurfMatcher:
    """
    A class to perform scale-invariant template matching using SURF features
    and FlannBasedMatcher. It can be initialized once and then used to match
    different template/target image pairs.

    Note: SURF is patented and might require a specific OpenCV build
    (opencv-contrib-python) and may have licensing restrictions for
    commercial use. This implementation assumes non-commercial, personal use.
    """

    def __init__(self, hessian_threshold=HESSIAN_THRESHOLD, lowe_ratio=LOWE_RATIO, min_match_count=MIN_MATCH_COUNT):
        """
        Initializes the SURF detector and FLANN matcher.

        Args:
            hessian_threshold (int): Threshold for SURF keypoint detection.
            lowe_ratio (float): Threshold for Lowe's ratio test for good matches.
            min_match_count (int): Minimum number of good matches required.

        Raises:
            AttributeError: If the required SURF module is not found in the
                            current OpenCV installation (likely needs opencv-contrib-python).
        """
        self.hessian_threshold = hessian_threshold
        self.lowe_ratio = lowe_ratio
        self.min_match_count = min_match_count

        # --- Initialize SURF Detector ---
        try:
            # Check if xfeatures2d is available
            if not hasattr(cv2, 'xfeatures2d'):
                 raise AttributeError("cv2.xfeatures2d module not found. Install 'opencv-contrib-python'.")
            self.surf = cv2.xfeatures2d.SURF_create(self.hessian_threshold)
            print(f"Using SURF with Hessian Threshold: {self.hessian_threshold}")
        except AttributeError as e:
            print(f"\nError initializing SURF: {e}")
            print("You might need to install 'opencv-contrib-python' (pip install opencv-contrib-python)")
            print("OR rebuild OpenCV with non-free modules enabled.")
            print("Ensure you have the correct version compatible with your base OpenCV.")
            raise AttributeError("SURF algorithm not available or failed to initialize.")

        # --- Initialize FLANN Matcher ---
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50) # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        print("SURF detector and FLANN matcher initialized.")


    def _load_image(self, image_input, mode=cv2.IMREAD_COLOR):
        """Loads an image from a path or uses the provided NumPy array."""
        if isinstance(image_input, str): # Check if it's a file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img = cv2.imread(image_input, mode)
            if img is None:
                raise Exception(f"Could not load image from: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray): # Check if it's already a NumPy array
            # If a color image is expected but grayscale is provided, convert it
            if mode == cv2.IMREAD_COLOR and len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            # If a grayscale image is expected but color is provided, convert it
            elif mode == cv2.IMREAD_GRAYSCALE and len(image_input.shape) == 3:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            return image_input # Assume it's in the correct format
        else:
            raise TypeError("Input must be a file path (str) or a NumPy array.")

    def match(self, template_input, target_input, show_visualization=SHOW_MATCH_VISUALIZATION):
        """
        Finds the template in the target image, draws a bounding box (clipped
        to image bounds), and optionally visualizes matches.

        Args:
            template_input (str | numpy.ndarray): File path or NumPy array of the template image.
            target_input (str | numpy.ndarray): File path or NumPy array of the target image.
            show_visualization (bool): If True, displays a window showing good matches.

        Returns:
            tuple: A tuple containing:
                - result_image (numpy.ndarray): The target image (in color) with
                  a bounding box drawn around the detected template (clipped to
                  image bounds), or the original color target image if not found.
                - homography_matrix (numpy.ndarray or None): The 3x3 perspective
                  transformation matrix if found, otherwise None.
                - clipped_corners (numpy.ndarray or None): The coordinates of the
                  template corners in the target image (clipped to image bounds)
                  if found, otherwise None.
                - status (str): A message indicating "Detected", "Not Detected",
                  or specific failure reason.

        Raises:
            FileNotFoundError: If an image path does not exist.
            Exception: If an image cannot be loaded or has insufficient features.
            TypeError: If inputs are not paths or NumPy arrays.
        """
        # --- Load Images ---
        template_img_gray = self._load_image(template_input, cv2.IMREAD_GRAYSCALE)
        target_img_color = self._load_image(target_input, cv2.IMREAD_COLOR)
        target_img_gray = cv2.cvtColor(target_img_color, cv2.COLOR_BGR2GRAY)
        target_h, target_w = target_img_color.shape[:2] # Get target dimensions for clipping

        # --- Detect Features ---
        kp_template, desc_template = self.surf.detectAndCompute(template_img_gray, None)
        kp_target, desc_target = self.surf.detectAndCompute(target_img_gray, None)

        # --- Validate Features ---
        if desc_template is None or len(kp_template) < 2:
            print(f"Warning: Not enough features found in template image ({len(kp_template)} found).")
            return target_img_color, None, None, "Not Detected (Insufficient template features)"
        if desc_target is None or len(kp_target) < 2:
            print(f"Warning: Not enough features found in target image ({len(kp_target)} found).")
            return target_img_color, None, None, "Not Detected (Insufficient target features)"

        # --- Match descriptors using FLANN ---
        # Ensure descriptors are float32, as required by FLANN
        if desc_template.dtype != np.float32:
            desc_template = np.float32(desc_template)
        if desc_target.dtype != np.float32:
            desc_target = np.float32(desc_target)

        matches = self.flann.knnMatch(desc_template, desc_target, k=2)

        # --- Filter matches using Lowe's ratio test ---
        good_matches = []
        for i, m_n in enumerate(matches):
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)
            elif len(m_n) == 1:
                 pass # Exclude single matches for robustness

        print(f"Found {len(kp_template)} keypoints in template.")
        print(f"Found {len(kp_target)} keypoints in target.")
        print(f"Found {len(matches)} raw potential match pairs.")
        print(f"Found {len(good_matches)} good matches after ratio test.")

        # --- Optional: Visualize the good matches ---
        if show_visualization:
            try:
                img_matches_visualization = cv2.drawMatches(
                    template_img_gray, kp_template,
                    target_img_gray, kp_target,
                    good_matches, None,
                    matchColor=MATCH_COLOR,
                    singlePointColor=None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.namedWindow("Good Matches Visualization", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Good Matches Visualization", 800, 600)
                # Show the matches in a window
                cv2.imshow("Good Matches Visualization", img_matches_visualization)
                print("\n--- VISUALIZATION ---")
                print("Showing the 'good' matches. Press any key in the window to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow("Good Matches Visualization")
                print("--- Continuing with Homography ---")
            except cv2.error as e:
                print(f"Error during match visualization: {e}")
                # Continue without visualization if it fails

        # --- Find Homography if enough good matches exist ---
        homography_matrix = None
        clipped_corners = None
        status = "Not Detected"
        result_img_output = target_img_color.copy() # Work on a copy for drawing

        if len(good_matches) >= self.min_match_count:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if homography_matrix is not None:
                h, w = template_img_gray.shape[:2]
                template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, homography_matrix)

                if transformed_corners is not None:
                    corners_int = np.int32(transformed_corners.reshape(-1, 2))
                    clipped_corners_list = []
                    for x, y in corners_int:
                        clipped_x = max(0, min(x, target_w - 1))
                        clipped_y = max(0, min(y, target_h - 1))
                        clipped_corners_list.append([clipped_x, clipped_y])

                    clipped_corners = np.array(clipped_corners_list, dtype=np.int32).reshape(-1, 1, 2)
                    result_img_output = cv2.polylines(result_img_output, [clipped_corners], True, DRAW_COLOR, 3, cv2.LINE_AA)

                    status = "Detected"
                    inlier_count = np.sum(mask) if mask is not None else 0
                    print(f"Object detected. Homography found. RANSAC Inliers: {inlier_count}/{len(good_matches)}")
                else:
                     print("Warning: perspectiveTransform returned None. Cannot draw bounding box.")
                     status = "Not Detected (Transform failed)"
                     homography_matrix = None # Ensure consistency
                     clipped_corners = None

            else:
                print("Warning: Homography could not be computed (findHomography returned None).")
                status = "Not Detected (Homography failed)"
                homography_matrix = None
                clipped_corners = None
        else:
            print(f"Warning: Not enough good matches found - {len(good_matches)}/{self.min_match_count}")
            status = f"Not Detected (Insufficient good matches: {len(good_matches)}/{self.min_match_count})"
            homography_matrix = None
            clipped_corners = None

        return result_img_output, homography_matrix, clipped_corners, status

# Example Usage (can be removed or kept for testing)
if __name__ == "__main__":
    # Define paths relative to the script location or use absolute paths
    script_dir = os.path.dirname(__file__) # Get directory where the script is located
    template_path = os.path.join(script_dir, "template.png") # Example path
    target_path = os.path.join(script_dir, "target.png")     # Example path

    # --- Create dummy images if they don't exist ---
    if not os.path.exists(template_path):
        print(f"Creating dummy template image at: {template_path}")
        dummy_template = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(dummy_template, (20, 20), (80, 80), (255, 0, 0), -1) # Blue square
        cv2.imwrite(template_path, dummy_template)

    if not os.path.exists(target_path):
        print(f"Creating dummy target image at: {target_path}")
        dummy_target = np.zeros((500, 500, 3), dtype=np.uint8)
        # Place a slightly rotated/scaled version of the template
        pts = np.array([[150, 150], [140, 260], [260, 280], [270, 170]], dtype=np.int32)
        cv2.fillPoly(dummy_target, [pts], (255, 0, 0)) # Blue polygon
        cv2.imwrite(target_path, dummy_target)
    # --- End of dummy image creation ---


    try:
        # Initialize the matcher
        matcher = ScaleInvariantSurfMatcher(hessian_threshold=100) # Adjust threshold if needed

        # --- Option 1: Match using file paths ---
        print("\n--- Matching using file paths ---")
        result_img, H, corners, status = matcher.match(template_path, target_path, show_visualization=False)
        print(f"Match Status: {status}")
        if status == "Detected":
            print("Homography Matrix:\n", H)
            print("Detected Corners (clipped):\n", corners.reshape(-1, 2)) # Reshape for readability
            cv2.imshow("Detection Result (Paths)", result_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Detection Result (Paths)")
        else:
            cv2.imshow("Detection Result (Paths) - Not Found", result_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Detection Result (Paths) - Not Found")


        # --- Option 2: Match using pre-loaded NumPy arrays ---
        print("\n--- Matching using NumPy arrays ---")
        template_np = cv2.imread(template_path) # Load as color initially
        target_np = cv2.imread(target_path)

        if template_np is None or target_np is None:
             print("Error: Could not load images for NumPy array test.")
        else:
            result_img_np, H_np, corners_np, status_np = matcher.match(template_np, target_np, show_visualization=False)
            print(f"Match Status (NumPy): {status_np}")
            if status_np == "Detected":
                print("Homography Matrix (NumPy):\n", H_np)
                print("Detected Corners (NumPy, clipped):\n", corners_np.reshape(-1, 2))
                cv2.imshow("Detection Result (NumPy)", result_img_np)
                cv2.waitKey(0)
                cv2.destroyWindow("Detection Result (NumPy)")
            else:
                cv2.imshow("Detection Result (NumPy) - Not Found", result_img_np)
                cv2.waitKey(0)
                cv2.destroyWindow("Detection Result (NumPy) - Not Found")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except AttributeError as e:
         print(f"Error: {e}. Ensure 'opencv-contrib-python' is installed and compatible.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed