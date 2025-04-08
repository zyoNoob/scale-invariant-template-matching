# 🎯 Scale-Invariant Template Matching using OpenCV SURF

This Python project provides a reusable class for performing scale-invariant template matching using OpenCV's SURF (Speeded-Up Robust Features) algorithm and FLANN (Fast Library for Approximate Nearest Neighbors) based matching. It allows you to find occurrences of a smaller template image within a larger target image, even if the template is scaled or rotated differently in the target.

---

## 💪 Features

-   **🖼️ Scale & Rotation Invariant:** Detects the template image regardless of its size or orientation in the target image.
-   **✨ SURF Feature Detection:** Utilizes the robust SURF algorithm for keypoint detection and description (requires `opencv-contrib-python`).
-   **⚡ Fast Matching:** Employs the FLANN-based matcher for efficient feature matching between template and target.
-   **📦 Modular Class:** Encapsulated in a `ScaleInvariantSurfMatcher` class for easy integration into other projects.
-   ** Flexible Input:** Accepts template and target images as file paths or NumPy arrays.
-   **📊 Clear Results:** Returns the target image with a bounding box drawn around the detected template, the homography matrix, corner coordinates, and a status message.
-   **🔧 Configurable:** Allows adjustment of key parameters like Hessian threshold, Lowe's ratio, and minimum match count.

---

## 👷️ Requirements

-   Python >= 3.13 (as specified in `pyproject.toml`)
-   Linux (Tested, SURF/OpenCV dependencies might vary on other OS)
-   OpenCV Contrib Python (`opencv-contrib-python`) - **Crucially, this project requires the `contrib` modules for SURF.**
-   NumPy

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

Import the `ScaleInvariantSurfMatcher` class and use its `match` method.

```python
# main.py (Example Usage)
import cv2
import os
import numpy as np # Import NumPy
from template_matching import ScaleInvariantSurfMatcher

def main():
    print("Running scale-invariant template matching example...")

    # Define paths relative to the script location (or use absolute paths)
    script_dir = os.path.dirname(__file__)
    # Make sure these point to your actual image files
    template_path = os.path.join(script_dir, "assets/template.png")
    target_path = os.path.join(script_dir, "assets/target.png")

    # --- Ensure images exist ---
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return
    if not os.path.exists(target_path):
        print(f"Error: Target image not found at {target_path}")
        return
    # --- ---

    try:
        # 1. Initialize the matcher
        # Adjust parameters if needed: hessian_threshold, lowe_ratio, min_match_count
        matcher = ScaleInvariantSurfMatcher(hessian_threshold=100, lowe_ratio=0.75, min_match_count=10)

        # 2. Perform matching (using file paths in this example)
        # You can also pass NumPy arrays directly: matcher.match(template_np_array, target_np_array)
        print(f"\nMatching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}'")
        result_img, homography, corners, status = matcher.match(
            template_path,
            target_path,
            show_visualization=False # Set to True to see intermediate matches
        )

        # 3. Process results
        print(f"Match Status: {status}")
        if status == "Detected":
            print("Template found!")
            # homography: 3x3 perspective transformation matrix
            # corners: Coordinates of the template corners in the target image (clipped)
            print(f"Detected Corners (clipped):\n{corners.reshape(-1, 2)}")

            # Display the result image with the bounding box
            cv2.imshow("Detection Result", result_img)
            print("Displaying result image. Press any key to close.")
            cv2.waitKey(0)

        else:
            print(f"Template not found or matching failed: {status}")
            # Optionally display the original target if not found
            # target_img_color = cv2.imread(target_path)
            # cv2.imshow("Target Image (Template Not Found)", target_img_color)
            # cv2.waitKey(0)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure template and target images exist.")
    except AttributeError as e:
         # This might happen if opencv-contrib-python is not installed correctly or missing SURF
         print(f"Error: {e}. Is 'opencv-contrib-python' installed and compatible? Does it include SURF?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed

if __name__ == "__main__":
    main()

```

### `match` Method Return Values:

-   `result_image` (numpy.ndarray): The target image (color) with a green bounding box drawn around the detected template (if found). The box corners are clipped to the image boundaries. If not found, it's the original color target image.
-   `homography_matrix` (numpy.ndarray | None): The 3x3 perspective transformation matrix if the template is found, otherwise `None`.
-   `clipped_corners` (numpy.ndarray | None): The `[[x, y]]` coordinates of the template corners in the target image (clipped to image bounds) if found, otherwise `None`.
-   `status` (str): A message indicating the outcome: "Detected", "Not Detected (Reason)".

---

## ⚙️ Configuration

The `ScaleInvariantSurfMatcher` class and the script have several configurable parameters:

-   **`hessian_threshold`** (in `__init__`): Controls the sensitivity of the SURF detector. Higher values detect fewer, potentially more robust features. Default: `100`.
-   **`lowe_ratio`** (in `__init__`): Threshold for the ratio test to filter good matches. Lower values are stricter. Default: `0.75`.
-   **`min_match_count`** (in `__init__`): Minimum number of good matches required to consider the template found. Default: `10`.
-   **`show_visualization`** (in `match`): Set to `True` to display a window showing the raw feature matches before homography calculation (useful for debugging). Default: `False`.

---

## 🧠 How It Works

1.  **Load Images:** Template and target images are loaded (grayscale for feature detection, color for output).
2.  **Detect SURF Features:** Keypoints and descriptors are computed for both images using `cv2.xfeatures2d.SURF_create()`.
3.  **Match Features:** FLANN matcher (`cv2.FlannBasedMatcher`) finds potential matching descriptors between the template and target.
4.  **Filter Matches (Lowe's Ratio Test):** Filters out ambiguous matches based on the distance ratio between the two nearest neighbors.
5.  **Find Homography:** If enough good matches are found (`>= min_match_count`), `cv2.findHomography` with RANSAC calculates the perspective transformation matrix mapping template points to target points.
6.  **Transform Corners:** The corners of the template image are transformed using the homography matrix to find their corresponding locations in the target image.
7.  **Clip & Draw Bounding Box:** The transformed corner coordinates are clipped to the target image boundaries, and a polygon is drawn on the output image.

---

## ⚠️ Important Note on SURF

SURF is patented and is **not free for commercial use**. Its inclusion in OpenCV is typically part of the `opencv-contrib-python` package, not the main `opencv-python` package. Ensure you have the correct package installed and are aware of potential licensing implications if using this project for commercial purposes.

---

## 📜 License

MIT License

---
