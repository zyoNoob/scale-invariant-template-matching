import cv2
import os
import numpy as np # Import NumPy
from template_matching import ScaleInvariantSurfMatcher, CannyEdgeMatcher, ColorMatcher # Import all classes

def main():
    print("Running template matching examples...")

    # Define paths relative to the script location
    script_dir = os.path.dirname(__file__)
    # Ensure the assets directory exists or adjust paths as needed
    assets_dir = os.path.join(script_dir, "assets")
    template_path = os.path.join(assets_dir, "template.png")
    target_path = os.path.join(assets_dir, "target.png")

    # --- SURF Matcher Example ---
    print("\n--- Running SURF Matcher ---")
    try:
        # 1. Initialize the SURF matcher
        surf_matcher = ScaleInvariantSurfMatcher(hessian_threshold=50)

        # 2. Perform matching using file paths
        print(f"Matching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}' (SURF)")
        result_img_surf, homography, corners, status_surf = surf_matcher.match(template_path, target_path, show_visualization=False)

        # 3. Process SURF results
        print(f"SURF Match Status: {status_surf}")
        if status_surf == "Detected":
            print("SURF: Template found!")
            # print("SURF Homography Matrix:\n", homography)
            # print("SURF Detected Corners (clipped):\n", corners.reshape(-1, 2))

            # Display the result
            cv2.imshow("SURF Detection Result", result_img_surf)
            print("Displaying SURF result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow("SURF Detection Result")
        else:
            print(f"SURF: Template not found or matching failed: {status_surf}")
            # Optionally display the target image even if not found
            # cv2.imshow("SURF Detection Result - Not Found", result_img_surf)
            # cv2.waitKey(0)
            # cv2.destroyWindow("SURF Detection Result - Not Found")

    except FileNotFoundError as e:
        print(f"Error during SURF matching: {e}. Please ensure template and target images exist.")
    except AttributeError as e:
         print(f"Error during SURF matching: {e}. Is 'opencv-contrib-python' installed and compatible?")
    except Exception as e:
        print(f"An unexpected error occurred during SURF matching: {e}")

    # --- Canny Edge Matcher Example (Multi-Scale) ---
    print("\n--- Running Canny Edge Matcher (Multi-Scale) ---")
    try:
        # 1. Initialize the Canny Edge matcher
        canny_matcher = CannyEdgeMatcher()

        # 2. Perform multi-scale matching (scale=None is default)
        print(f"Matching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}' (Canny Edge, Multi-Scale)")
        result_img_canny_multi, bbox_canny_multi, scale_canny_multi, corr_canny_multi, status_canny_multi = canny_matcher.match(template_path, target_path)

        # 3. Process Multi-Scale Canny Edge results
        print(f"Canny Edge Match Status (Multi-Scale): {status_canny_multi}")
        if status_canny_multi == "Detected":
            print("Canny Edge (Multi-Scale): Template found!")
            print(f"Canny Bounding Box (x, y, w, h): {bbox_canny_multi}")
            print(f"Canny Best Scale Found: {scale_canny_multi:.2f}")
            print(f"Canny Best Correlation: {corr_canny_multi:.4f}")

            # Display the result
            cv2.imshow("Canny Edge Detection Result (Multi-Scale)", result_img_canny_multi)
            print("Displaying Canny Edge (Multi-Scale) result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow("Canny Edge Detection Result (Multi-Scale)")
        else:
            print(f"Canny Edge (Multi-Scale): Template not found. Best correlation: {corr_canny_multi:.4f}")
            # Optionally display the target image even if not found
            # cv2.imshow("Canny Edge Detection Result (Multi-Scale) - Not Found", result_img_canny_multi)
            # cv2.waitKey(0)
            # cv2.destroyWindow("Canny Edge Detection Result (Multi-Scale) - Not Found")

    except FileNotFoundError as e:
        print(f"Error during Canny Edge (Multi-Scale) matching: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Canny Edge (Multi-Scale) matching: {e}")

    # --- Canny Edge Matcher Example (Fixed Scale) ---
    print("\n--- Running Canny Edge Matcher (Fixed Scale) ---")
    try:
        # 1. Re-use or re-initialize the matcher (initialization is cheap)
        canny_matcher_fixed = CannyEdgeMatcher() # Scales params not needed here

        # 2. Perform fixed-scale matching - choose a scale to test (e.g., 1.0 or a known good scale)
        fixed_scale_to_test = scale_canny_multi if scale_canny_multi else 1.1 # Example scale
        print(f"Matching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}' (Canny Edge, Fixed Scale={fixed_scale_to_test})")
        result_img_canny_fixed, bbox_canny_fixed, scale_canny_fixed, corr_canny_fixed, status_canny_fixed = canny_matcher_fixed.match(
            template_path, target_path, scale=fixed_scale_to_test
        )

        # 3. Process Fixed-Scale Canny Edge results
        print(f"Canny Edge Match Status (Fixed Scale): {status_canny_fixed}")
        if status_canny_fixed == "Detected":
            print(f"Canny Edge (Fixed Scale {fixed_scale_to_test}): Template found!")
            print(f"Canny Bounding Box (x, y, w, h): {bbox_canny_fixed}")
            # Note: scale_canny_fixed should match fixed_scale_to_test if detected
            print(f"Canny Scale Used: {scale_canny_fixed:.2f}")
            print(f"Canny Correlation: {corr_canny_fixed:.4f}")

            # Display the result
            cv2.imshow(f"Canny Edge Detection Result (Fixed Scale {fixed_scale_to_test})", result_img_canny_fixed)
            print(f"Displaying Canny Edge (Fixed Scale {fixed_scale_to_test}) result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow(f"Canny Edge Detection Result (Fixed Scale {fixed_scale_to_test})")
        else:
            print(f"Canny Edge (Fixed Scale {fixed_scale_to_test}): Template not found. Correlation: {corr_canny_fixed:.4f}")
            # Optionally display the target image even if not found
            # cv2.imshow(f"Canny Edge Detection Result (Fixed Scale {fixed_scale_to_test}) - Not Found", result_img_canny_fixed)
            # cv2.waitKey(0)
            # cv2.destroyWindow(f"Canny Edge Detection Result (Fixed Scale {fixed_scale_to_test}) - Not Found")

    except FileNotFoundError as e:
        print(f"Error during Canny Edge (Fixed Scale) matching: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Canny Edge (Fixed Scale) matching: {e}")

    # --- Color Matcher Example (Multi-Scale) ---
    print("\n--- Running Color Matcher (Multi-Scale) ---")
    try:
        # 1. Initialize the Color matcher
        # Adjust threshold, scales etc. as needed
        color_matcher = ColorMatcher(match_threshold=0.7, num_scales=50, min_scale=0.7, max_scale=1.3)

        # 2. Perform multi-scale matching
        print(f"Matching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}' (Color, Multi-Scale)")
        result_img_color_multi, bbox_color_multi, scale_color_multi, corr_color_multi, status_color_multi = color_matcher.match(template_path, target_path)

        # 3. Process Multi-Scale Color results
        print(f"Color Match Status (Multi-Scale): {status_color_multi}")
        if status_color_multi == "Detected":
            print("Color (Multi-Scale): Template found!")
            print(f"Color Bounding Box (x, y, w, h): {bbox_color_multi}")
            print(f"Color Best Scale Found: {scale_color_multi:.2f}")
            print(f"Color Best Correlation: {corr_color_multi:.4f}")

            # Display the result
            cv2.imshow("Color Detection Result (Multi-Scale)", result_img_color_multi)
            print("Displaying Color (Multi-Scale) result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow("Color Detection Result (Multi-Scale)")
        else:
            print(f"Color (Multi-Scale): Template not found. Best correlation: {corr_color_multi:.4f}")

    except FileNotFoundError as e:
        print(f"Error during Color (Multi-Scale) matching: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Color (Multi-Scale) matching: {e}")

    # --- Color Matcher Example (Fixed Scale) ---
    print("\n--- Running Color Matcher (Fixed Scale) ---")
    try:
        # 1. Re-use or re-initialize the matcher
        color_matcher_fixed = ColorMatcher(match_threshold=0.7) # Scales params not needed here

        # 2. Perform fixed-scale matching - use scale found previously or a default
        fixed_scale_color_test = scale_color_multi if 'scale_color_multi' in locals() and scale_color_multi else 1.0 # Example scale
        print(f"Matching template '{os.path.basename(template_path)}' in target '{os.path.basename(target_path)}' (Color, Fixed Scale={fixed_scale_color_test})")
        result_img_color_fixed, bbox_color_fixed, scale_color_fixed, corr_color_fixed, status_color_fixed = color_matcher_fixed.match(
            template_path, target_path, scale=fixed_scale_color_test
        )

        # 3. Process Fixed-Scale Color results
        print(f"Color Match Status (Fixed Scale): {status_color_fixed}")
        if status_color_fixed == "Detected":
            print(f"Color (Fixed Scale {fixed_scale_color_test}): Template found!")
            print(f"Color Bounding Box (x, y, w, h): {bbox_color_fixed}")
            print(f"Color Scale Used: {scale_color_fixed:.2f}")
            print(f"Color Correlation: {corr_color_fixed:.4f}")

            # Display the result
            cv2.imshow(f"Color Detection Result (Fixed Scale {fixed_scale_color_test})", result_img_color_fixed)
            print(f"Displaying Color (Fixed Scale {fixed_scale_color_test}) result image. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow(f"Color Detection Result (Fixed Scale {fixed_scale_color_test})")
        else:
            print(f"Color (Fixed Scale {fixed_scale_color_test}): Template not found. Correlation: {corr_color_fixed:.4f}")

    except FileNotFoundError as e:
        print(f"Error during Color (Fixed Scale) matching: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during Color (Fixed Scale) matching: {e}")

    finally:
        print("\nFinished examples. Closing any remaining OpenCV windows.")
        cv2.destroyAllWindows() # Ensure all OpenCV windows are closed

if __name__ == "__main__":
    main()