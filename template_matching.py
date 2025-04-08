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
        self.hessian_threshold = int(hessian_threshold)
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

class CannyEdgeMatcher:
    """
    Performs multi-scale template matching using Canny edge detection.
    It iterates through different scales of the template image to find the best match.
    """
    def __init__(self, canny_low=0, canny_high=200, num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.25):
        """
        Initializes the Canny Edge Matcher.

        Args:
            canny_low (int): Lower threshold for the Canny edge detector.
            canny_high (int): Higher threshold for the Canny edge detector.
            num_scales (int): Number of scales to check between min_scale and max_scale.
            min_scale (float): The minimum scale factor for the template.
            max_scale (float): The maximum scale factor for the template.
            match_threshold (float): Minimum correlation coefficient (from TM_CCOEFF_NORMED)
                                     to consider a match valid.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.num_scales = num_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.match_threshold = match_threshold
        print(f"CannyEdgeMatcher initialized with thresholds ({canny_low}, {canny_high}), "
              f"scale range [{min_scale}-{max_scale}] ({num_scales} steps), "
              f"match threshold {match_threshold}")

    def _load_image(self, image_input, mode=cv2.IMREAD_GRAYSCALE):
        """Loads an image from a path or uses the provided NumPy array (grayscale)."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img = cv2.imread(image_input, mode)
            if img is None:
                raise Exception(f"Could not load image from: {image_input}")
            return img
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and mode == cv2.IMREAD_GRAYSCALE:
                return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif len(image_input.shape) == 2 and mode == cv2.IMREAD_COLOR:
                 return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input # Assume correct format otherwise
        else:
            raise TypeError("Input must be a file path (str) or a NumPy array.")

    def match(self, template_input, target_input, scale=None, draw_result=True):
        """
        Finds the template in the target image using Canny edge matching.
        If 'scale' is provided, it matches only at that scale. Otherwise, it
        performs multi-scale matching.

        Args:
            template_input (str | numpy.ndarray): File path or NumPy array of the template image.
            target_input (str | numpy.ndarray): File path or NumPy array of the target image.
            scale (float, optional): If provided, performs matching only at this specific scale factor.
                                     Defaults to None (multi-scale matching).
            draw_result (bool): If True, draws the bounding box on the result image.

        Returns:
            tuple: A tuple containing:
                - result_image (numpy.ndarray): The target image (in color) with
                  a bounding box drawn if found and draw_result is True.
                - bounding_box (tuple or None): (x, y, w, h) of the best match, or None.
                - best_scale (float or None): The scale factor of the best match, or None.
                - max_correlation (float): The correlation value of the best match found.
                - status (str): "Detected" or "Not Detected".
        """
        template_gray = self._load_image(template_input, cv2.IMREAD_GRAYSCALE)
        target_color = self._load_image(target_input, cv2.IMREAD_COLOR) # Load color for drawing
        target_gray = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)

        if template_gray is None or target_gray is None:
            return target_color, None, None, 0.0, "Error loading images"

        (tH, tW) = template_gray.shape[:2]

        # Compute Canny edges for the target image (only once)
        target_edges = cv2.Canny(target_gray, self.canny_low, self.canny_high)

        best_match = None # Store (max_val, max_loc, scale, width, height)

        # Determine scales to check
        if scale is not None:
            scales_to_check = [scale]
            print(f"Performing Canny Edge match at fixed scale: {scale:.2f}")
        else:
            scales_to_check = np.linspace(self.min_scale, self.max_scale, self.num_scales)[::-1]
            print("Performing multi-scale Canny Edge match...")


        # Loop over scales of the template (could be single or multiple scales)
        for current_scale in scales_to_check:
            # Calculate new dimensions based on scale, ensuring they are integers > 0
            new_w = max(1, int(tW * current_scale))
            new_h = max(1, int(tH * current_scale))
            # Use cv2.resize
            resized_template = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            (rH, rW) = resized_template.shape[:2]

            # Check if resized template is larger than the target
            if rW > target_gray.shape[1] or rH > target_gray.shape[0]:
                # If using a fixed scale and it's too large, report and exit loop
                if scale is not None:
                    print(f"Warning: Provided scale {scale:.2f} results in template ({rW}x{rH}) larger than target ({target_gray.shape[1]}x{target_gray.shape[0]}).")
                    best_match = None # Ensure no match is reported
                    break # Exit loop
                else:
                    continue # Skip this scale in multi-scale search

            # Compute Canny edges for the scaled template
            template_edges = cv2.Canny(resized_template, self.canny_low, self.canny_high)

            # Perform template matching
            try:
                # Ensure template edges are not empty and smaller than target edges
                if template_edges.size == 0 or template_edges.shape[0] > target_edges.shape[0] or template_edges.shape[1] > target_edges.shape[1]:
                    # print(f"Skipping scale {current_scale:.2f}: Invalid template edge dimensions.")
                    continue

                result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)
                (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

                # Update best match if current match is better (or if it's the only one)
                if best_match is None or max_val > best_match[0]:
                    best_match = (max_val, max_loc, current_scale, rW, rH)
                    # If using fixed scale, we can break after the first successful match attempt
                    # (or keep going if we only want the result from this specific scale regardless of others)
                    # For simplicity, we let it complete the single iteration.

            except cv2.error as e:
                # print(f"Skipping scale {current_scale:.2f} due to cv2.error in matchTemplate: {e}")
                # If using fixed scale and it errors, report and exit loop
                if scale is not None:
                    print(f"Error matching at fixed scale {scale:.2f}: {e}")
                    best_match = None # Ensure no match is reported
                    break # Exit loop
                else:
                    continue # Skip this scale in multi-scale search

        # Process the best match found (if any)
        result_img_output = target_color.copy()
        bounding_box = None
        best_scale_found = None # Use a different name to avoid confusion with input 'scale'
        max_correlation = 0.0
        status = "Not Detected"

        if best_match is not None and best_match[0] >= self.match_threshold:
            max_correlation, max_loc, best_scale_found, w, h = best_match
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            bounding_box = (top_left[0], top_left[1], w, h)
            status = "Detected"
            print(f"Object detected with Canny Edges. Correlation: {max_correlation:.4f} at scale {best_scale_found:.2f}")

            if draw_result:
                cv2.rectangle(result_img_output, top_left, bottom_right, DRAW_COLOR, 2)
        elif best_match is not None: # Match found but below threshold
             max_correlation = best_match[0]
             best_scale_found = best_match[2]
             print(f"Template not detected (correlation {max_correlation:.4f} < threshold {self.match_threshold}). Best match was at scale {best_scale_found:.2f}.")
        else: # No match found at all (e.g., scale too large or errored)
             print(f"Template not detected. No valid matches found across checked scales.")


        return result_img_output, bounding_box, best_scale_found, max_correlation, status