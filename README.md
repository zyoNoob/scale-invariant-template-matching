# 🎯 Scale-Invariant Template Matching using OpenCV

This repository provides Python classes for performing template matching in images, designed to be robust against changes in scale. It includes three main approaches:

1.  **`ScaleInvariantSurfMatcher`**: Uses SURF (Speeded-Up Robust Features) for feature detection and matching, combined with FLANN (Fast Library for Approximate Nearest Neighbors) and homography estimation. This method is generally robust to rotation and perspective changes in addition to scale.
    *   **Note:** SURF is patented and might require installing `opencv-contrib-python` and may have licensing restrictions for commercial use.
2.  **`CannyEdgeMatcher`**: Uses Canny edge detection on both the template and target images, then performs multi-scale template matching (`cv2.matchTemplate`) by resizing the *template* across a range of scales. This method is simpler and does not require `opencv-contrib-python`, but is primarily designed for scale invariance and may be less robust to significant rotation or perspective distortion compared to SURF. It also supports matching at a single, pre-defined scale.
3.  **`ColorMatcher`**: Uses color images and multi-scale template matching (`cv2.TM_CCOEFF_NORMED`). Similar to `CannyEdgeMatcher` but operates directly on color information. Does not require `contrib` modules.

---

## 💪 Features

-   **🖼️ Scale & Rotation Invariant:** Detects the template image regardless of its size or orientation in the target image.
-   **✨ SURF Feature Detection:** Utilizes the robust SURF algorithm for keypoint detection and description (requires `opencv-contrib-python`).
-   **⚡ Fast Matching:** Employs the FLANN-based matcher for efficient feature matching between template and target.
-   **📦 Modular Class:** Encapsulated in a `ScaleInvariantSurfMatcher` class for easy integration into other projects.
-   ** Flexible Input:** Accepts template and target images as file paths or NumPy arrays.
-   **📊 Clear Results:** Returns the target image with a bounding box drawn around the detected template, the homography matrix, corner coordinates, and a status message.
-   **🔧 Configurable:** Allows adjustment of key parameters like Hessian threshold, Lowe's ratio, and minimum match count.
-   **Canny Edge Method:**
    *   Uses edge information for matching.
    *   Efficient multi-scale search by resizing the template.
    *   Option to match at a single, fixed scale.
    *   Returns a simple bounding box (`x, y, w, h`) and the best scale found.
-   **Color Method:**
    *   Uses direct color template matching (`TM_CCOEFF_NORMED`).
    *   Efficient multi-scale search (resizes template).
    *   Option for fixed-scale matching.
    *   Returns bounding box and best scale.

---

## 👷️ Requirements

-   Python >= 3.13 (as specified in `pyproject.toml`)
-   Linux (Tested, SURF/OpenCV dependencies might vary on other OS)
-   OpenCV Contrib Python (`opencv-contrib-python`) - **Crucially, this project requires the `contrib` modules for SURF.**
-   NumPy
-   **For `CannyEdgeMatcher` and `ColorMatcher`:** No extra OpenCV modules required beyond standard `opencv-python`.

---

## 🛠️ Installation & Setup

This project uses `uv` for dependency management.

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/zyoNoob/scale-invariant-template-matching.git
    cd scale-invariant-template-matching
    ```

2.  **Install dependencies using `uv`:**
    The `pyproject.toml` is configured to potentially use a specific wheel file for `opencv-contrib-python` located in the `whl/` directory. `uv` should handle this automatically.
    ```bash
    uv sync
    ```
    *If `uv` is not installed, you can install it following its official instructions.*
    *Alternatively, if you manage dependencies manually, ensure you install `numpy` and the correct version of `opencv-contrib-python` for your Python version and OS.*

3.  **Ensure Image Files Exist:**
    The example code uses `template.png` and `target.png`. Sample images are provided in the `assets/` directory. You can replace these with your own images.

---

## 🚀 Usage

Import the desired matcher class(es) from `template_matching.py` and use their `match` methods.

```python
# Example snippet from main.py

import cv2
from template_matching import ScaleInvariantSurfMatcher, CannyEdgeMatcher, ColorMatcher

# --- Define image paths ---
template_path = "assets/template.png"
target_path = "assets/target.png"

# --- Using ScaleInvariantSurfMatcher ---
try:
    surf_matcher = ScaleInvariantSurfMatcher(hessian_threshold=100)
    result_img_surf, H, corners, status_surf = surf_matcher.match(template_path, target_path)
    if status_surf == "Detected":
        print("SURF: Template Found!")
        # cv2.imshow("SURF Result", result_img_surf)
        # cv2.waitKey(0)
except Exception as e:
    print(f"SURF Error: {e}")


# --- Using CannyEdgeMatcher (Multi-Scale) ---
try:
    canny_matcher = CannyEdgeMatcher(match_threshold=0.4, num_scales=50, min_scale=0.5, max_scale=2.0)
    result_img_canny, bbox, scale, correlation, status_canny = canny_matcher.match(template_path, target_path)
    if status_canny == "Detected":
        print(f"Canny (Multi-Scale): Template Found at scale {scale:.2f}!")
        # cv2.imshow("Canny Multi-Scale Result", result_img_canny)
        # cv2.waitKey(0)
except Exception as e:
    print(f"Canny Multi-Scale Error: {e}")

# --- Using CannyEdgeMatcher (Fixed Scale) ---
try:
    # Matcher can be reused or re-initialized
    fixed_scale = 1.2 # Example scale
    result_img_canny_fixed, bbox_fixed, _, correlation_fixed, status_canny_fixed = canny_matcher.match(
        template_path, target_path, scale=fixed_scale
    )
    if status_canny_fixed == "Detected":
        print(f"Canny (Fixed Scale {fixed_scale}): Template Found!")
        # cv2.imshow("Canny Fixed Scale Result", result_img_canny_fixed)
        # cv2.waitKey(0)
except Exception as e:
    print(f"Canny Fixed Scale Error: {e}")

# --- Using ColorMatcher (Multi-Scale) ---
try:
    color_matcher = ColorMatcher(match_threshold=0.7)
    result_img_color, bbox_color, scale_color, corr_color, status_color = color_matcher.match(template_path, target_path)
    if status_color == "Detected":
        print(f"Color (Multi-Scale): Template Found at scale {scale_color:.2f}!")
        # cv2.imshow("Color Multi-Scale Result", result_img_color)
        # cv2.waitKey(0)
except Exception as e:
    print(f"Color Multi-Scale Error: {e}")

# --- Using ColorMatcher (Fixed Scale) ---
try:
    fixed_scale_color = 1.0 # Example scale
    result_img_color_fixed, bbox_color_fixed, _, corr_color_fixed, status_color_fixed = color_matcher.match(
        template_path, target_path, scale=fixed_scale_color
    )
    if status_color_fixed == "Detected":
        print(f"Color (Fixed Scale {fixed_scale_color}): Template Found!")
        # cv2.imshow("Color Fixed Scale Result", result_img_color_fixed)
        # cv2.waitKey(0)
except Exception as e:
    print(f"Color Fixed Scale Error: {e}")


cv2.destroyAllWindows()

```

### `match` Method Return Values:

#### `ScaleInvariantSurfMatcher`

-   `result_image` (numpy.ndarray): The target image (color) with a green bounding box drawn around the detected template (if found). The box corners are clipped to the image boundaries. If not found, it's the original color target image.
-   `homography_matrix` (numpy.ndarray | None): The 3x3 perspective transformation matrix if the template is found, otherwise `None`.
-   `clipped_corners` (numpy.ndarray | None): The `[[x, y]]` coordinates of the template corners in the target image (clipped to image bounds) if found, otherwise `None`.
-   `status` (str): A message indicating the outcome: "Detected", "Not Detected (Reason)".

#### `CannyEdgeMatcher`

-   `result_image` (numpy.ndarray): The target image (color) with a bounding box drawn around the detected template (if found). If not found, it's the original color target image.
-   `bounding_box` (tuple | None): Tuple `(x, y, w, h)` of the best match found, or `None`.
-   `best_scale` (float | None): The scale factor at which the best match was found, or `None`.
-   `max_correlation` (float): The correlation score (`TM_CCOEFF_NORMED`) of the best match found (even if below threshold).
-   `status` (str): A message indicating the outcome: "Detected", "Not Detected (Reason)".

#### `ColorMatcher`

-   `result_image` (numpy.ndarray): The target image (color) with a bounding box drawn around the detected template (if found). If not found, it's the original color target image.
-   `bounding_box` (tuple | None): Tuple `(x, y, w, h)` of the best match found, or `None`.
-   `best_scale` (float | None): The scale factor at which the best match was found, or `None`.
-   `max_correlation` (float): The correlation score (`TM_CCOEFF_NORMED`) of the best match found (even if below threshold).
-   `status` (str): A message indicating the outcome: "Detected", "Not Detected".

---

## ⚙️ Configuration

The `ScaleInvariantSurfMatcher`, `CannyEdgeMatcher`, and `ColorMatcher` classes have several configurable parameters:

### `ScaleInvariantSurfMatcher`

-   **`hessian_threshold`** (in `__init__`): Controls the sensitivity of the SURF detector. Higher values detect fewer, potentially more robust features. Default: `100`.
-   **`lowe_ratio`** (in `__init__`): Threshold for the ratio test to filter good matches. Lower values are stricter. Default: `0.75`.
-   **`min_match_count`** (in `__init__`): Minimum number of good matches required to consider the template found. Default: `10`.
-   **`show_visualization`** (in `match`): Set to `True` to display a window showing the raw feature matches before homography calculation (useful for debugging). Default: `False`.

### `CannyEdgeMatcher`

*   **`__init__(self, canny_low=0, canny_high=100, num_scales=150, min_scale=0.5, max_scale=2.0, match_threshold=0.6)`**
    *   Initializes the Canny edge matcher parameters.
    *   `canny_low`, `canny_high`: Thresholds for the Canny edge detector applied to both template (at each scale) and target.
    *   `num_scales`: Number of scales to check between `min_scale` and `max_scale` (only used if `match` is called without a specific `scale`).
    *   `min_scale`, `max_scale`: Range of scaling factors applied to the template (only used if `match` is called without a specific `scale`).
    *   `match_threshold`: Minimum correlation coefficient from `cv2.matchTemplate` (using `TM_CCOEFF_NORMED`) needed to consider a detection valid.
*   **`match(self, template_input, target_input, scale=None, draw_result=True)`**
    *   Performs the matching process using Canny edges.
    *   `template_input`, `target_input`: File paths (str) or NumPy arrays.
    *   `scale` (float, optional): If provided, matching is performed *only* at this specific scale factor, ignoring `num_scales`, `min_scale`, `max_scale`. If `None` (default), multi-scale matching is performed using the parameters from `__init__`.
    *   `draw_result`: If `True`, draws the bounding box on the result image.
    *   **Returns:** `(result_image, bounding_box, best_scale, max_correlation, status)`
        *   `result_image`: Target image (color) with bounding box drawn if detected and `draw_result` is True.
        *   `bounding_box`: Tuple `(x, y, w, h)` of the best match found, or `None`.
        *   `best_scale`: The scale factor at which the best match was found, or `None`.
        *   `max_correlation`: The correlation score (`TM_CCOEFF_NORMED`) of the best match found (even if below threshold).
        *   `status`: String indicating result ("Detected", "Not Detected").

### `ColorMatcher`

*   **`__init__(self, num_scales=50, min_scale=0.5, max_scale=1.5, match_threshold=0.7)`**
    *   Initializes the Color matcher parameters.
    *   `num_scales`: Number of scales to check (only used for multi-scale matching).
    *   `min_scale`, `max_scale`: Range of scaling factors (only used for multi-scale matching).
    *   `match_threshold`: Minimum correlation coefficient (`TM_CCOEFF_NORMED`) for a valid detection.
*   **`match(self, template_input, target_input, scale=None, draw_result=True)`**
    *   Performs the matching process using color images.
    *   `scale` (float, optional): If provided, performs matching only at this specific scale factor. If `None` (default), multi-scale matching is performed.
    *   `draw_result`: If `True`, draws the bounding box on the result image.
    *   **Returns:** `(result_image, bounding_box, best_scale, max_correlation, status)` (See Return Values section above).

---

## 🧠 How It Works

### `ScaleInvariantSurfMatcher`

1.  **Load Images:** Template and target images are loaded (grayscale for feature detection, color for output).
2.  **Detect SURF Features:** Keypoints and descriptors are computed for both images using `cv2.xfeatures2d.SURF_create()`.
3.  **Match Features:** FLANN matcher (`cv2.FlannBasedMatcher`) finds potential matching descriptors between the template and target.
4.  **Filter Matches (Lowe's Ratio Test):** Filters out ambiguous matches based on the distance ratio between the two nearest neighbors.
5.  **Find Homography:** If enough good matches are found (`>= min_match_count`), `cv2.findHomography` with RANSAC calculates the perspective transformation matrix mapping template points to target points.
6.  **Transform Corners:** The corners of the template image are transformed using the homography matrix to find their corresponding locations in the target image.
7.  **Clip & Draw Bounding Box:** The transformed corner coordinates are clipped to the target image boundaries, and a polygon is drawn on the output image.

### `CannyEdgeMatcher`

1.  **Load Images:** Template and target images are loaded (grayscale for edge detection, color for output).
2.  **Apply Canny Edge Detection:** Edges are detected in both images using the specified thresholds.
3.  **Multi-Scale Matching:** The template is resized across a range of scales, and `cv2.matchTemplate` is used to find the best match at each scale.
4.  **Fixed-Scale Matching:** If a specific scale is provided, matching is performed only at that scale.
5.  **Find Best Match:** The scale and bounding box with the highest correlation score are selected.
6.  **Draw Bounding Box:** The bounding box is drawn on the output image.

### `ColorMatcher`

1.  **Load Images:** Template and target images are loaded in color (BGR format). Alpha channels are removed if present.
2.  **Multi-Scale/Fixed-Scale Loop:**
    *   If `scale` is provided in `match()`, only that scale is used.
    *   Otherwise, iterate through scales from `max_scale` down to `min_scale`.
3.  **Resize Template:** The color template image is resized to the current scale.
4.  **Template Matching:** `cv2.matchTemplate` with `cv2.TM_CCOEFF_NORMED` compares the resized color template against the color target image.
5.  **Find Best Match:** Keep track of the scale and location (`max_loc`) that yield the highest correlation score (`max_val`).
6.  **Check Threshold & Draw Box:** If the best `max_val` is above `match_threshold`, the status is "Detected", and a bounding box corresponding to the template's size at the best scale is drawn on the output image (if `draw_result` is True).

---

## ⚠️ Important Note on SURF

SURF is patented and is **not free for commercial use**. Its inclusion in OpenCV is typically part of the `opencv-contrib-python` package, not the main `opencv-python` package. Ensure you have the correct package installed and are aware of potential licensing implications if using this project for commercial purposes.

---

## 📜 License

MIT License

---
