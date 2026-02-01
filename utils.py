import cv2
import numpy as np

# -----------------------
# 1. Resize
# -----------------------
def resize_image(img, width, height):
    return cv2.resize(img, (width, height))


# -----------------------
# 2. Rotate
# -----------------------
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))


# -----------------------
# 3. Flip
# -----------------------
def flip_image(img, flip_code):
    codes = {"Horizontal": 1, "Vertical": 0, "Both": -1}
    return cv2.flip(img, codes[flip_code])


# -----------------------
# 4. Color Conversion
# -----------------------
def convert_color(img, color_space):
    spaces = {"GRAY": cv2.COLOR_BGR2GRAY, "HSV": cv2.COLOR_BGR2HSV, "LAB": cv2.COLOR_BGR2LAB}
    return cv2.cvtColor(img, spaces[color_space])


# -----------------------
# 5. Extract RGB
# -----------------------
def get_rgb(img, x, y):
    if len(img.shape) == 2:  # grayscale
        return img[y, x], img[y, x], img[y, x]
    b, g, r = img[y, x]
    return r, g, b


# -----------------------
# 6. Draw Shapes
# -----------------------
def draw_shapes(img, shape_type, sidebar):
    img_copy = img.copy()
    if shape_type == "Line":
        x1 = sidebar.number_input("x1", 0, img.shape[1], 0)
        y1 = sidebar.number_input("y1", 0, img.shape[0], 0)
        x2 = sidebar.number_input("x2", 0, img.shape[1], 100)
        y2 = sidebar.number_input("y2", 0, img.shape[0], 100)
        color = (255, 0, 0)
        thickness = 2
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    elif shape_type == "Rectangle":
        x = sidebar.number_input("x", 0, img.shape[1], 0)
        y = sidebar.number_input("y", 0, img.shape[0], 0)
        w = sidebar.number_input("width", 1, img.shape[1], 100)
        h = sidebar.number_input("height", 1, img.shape[0], 100)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, thickness)
    elif shape_type == "Circle":
        x = sidebar.number_input("x", 0, img.shape[1], 50)
        y = sidebar.number_input("y", 0, img.shape[0], 50)
        radius = sidebar.number_input("radius", 1, 500, 50)
        color = (0, 0, 255)
        thickness = 2
        cv2.circle(img_copy, (x, y), radius, color, thickness)
    elif shape_type == "Text":
        x = sidebar.number_input("x", 0, img.shape[1], 50)
        y = sidebar.number_input("y", 0, img.shape[0], 50)
        text = sidebar.text_input("Text", "Hello")
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(img_copy, text, (x, y), font, 1, color, 2, cv2.LINE_AA)
    return img_copy


# -----------------------
# 7. Addition / Subtraction
# -----------------------
def add_subtract_images(img1, img2, operation):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if operation == "Add":
        return cv2.add(img1, img2)
    elif operation == "Subtract":
        return cv2.subtract(img1, img2)
    return img1


# -----------------------
# 8. Bitwise Operations
# -----------------------
def bitwise_op(img, mask, operation):
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if operation == "AND":
        return cv2.bitwise_and(img, img, mask=mask)
    elif operation == "OR":
        return cv2.bitwise_or(img, img, mask=mask)
    elif operation == "XOR":
        return cv2.bitwise_xor(img, img, mask=mask)
    elif operation == "NOT":
        return cv2.bitwise_not(img, mask=mask)
    return img


# -----------------------
# 9. Edge Detection
# -----------------------
def edge_detection(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Canny":
        return cv2.Canny(gray, 100, 200)
    elif method == "Sobel":
        return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    elif method == "Laplacian":
        return cv2.Laplacian(gray, cv2.CV_64F)
    return gray


# -----------------------
# 10. Thresholding
# -----------------------
def thresholding(img, th_type):
    # Convert to grayscale if needed
    if len(img.shape) == 2:  # already grayscale
        gray = img
    elif len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Ensure dtype is uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)

    # Apply threshold
    if th_type == "Simple":
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return th
    elif th_type == "Adaptive":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return th
    elif th_type == "Otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    return gray



# -----------------------
# 11. Blur
# -----------------------
def blur_image(img, ksize, method="Gaussian"):
    """
    Apply different blur methods to an image.
    Parameters:
        img    : input image (grayscale or BGR)
        ksize  : kernel size (must be odd)
        method : "Gaussian", "Median", "Average", "Bilateral"
    Returns:
        Blurred image
    """
    # Ensure kernel is odd
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    if method == "Gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif method == "Median":
        return cv2.medianBlur(img, ksize)
    elif method == "Average":
        return cv2.blur(img, (ksize, ksize))
    elif method == "Bilateral":
        # d = diameter of pixel neighborhood, sigmaColor and sigmaSpace
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown blur method: {method}")



# -----------------------
# 12. Object Counting
# -----------------------
import cv2
import numpy as np

def count_objects(img, draw_contours=False):
    """
    Count objects in an image.

    Parameters:
        img (numpy.ndarray): Input BGR image
        draw_contours (bool): If True, draw contours, bounding boxes, and areas on the image

    Returns:
        count (int): Number of objects detected
        output_img (numpy.ndarray): Image with contours drawn (if draw_contours=True)
    """
    # Make a copy of original image to draw on
    output_img = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours if requested
    if draw_contours:
        for cnt in contours:
            # Draw contour
            cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)

            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Put contour area text
            area = cv2.contourArea(cnt)
            cv2.putText(output_img, f'{int(area)}', (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # Return count and optionally drawn image
    return len(contours), output_img if draw_contours else img


# -----------------------
# 13. Morphological Operations
# -----------------------
def morphological_op(img, op_type, ksize):
    kernel = np.ones((ksize, ksize), np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if op_type == "Erode":
        return cv2.erode(gray, kernel, iterations=1)
    elif op_type == "Dilate":
        return cv2.dilate(gray, kernel, iterations=1)
    elif op_type == "Open":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif op_type == "Close":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray


# -----------------------
# 14. Feature Matching (ORB)
# -----------------------
def feature_matching(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    orb = cv2.ORB_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        return img1
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)


# -----------------------
# 15. Image Translation
# -----------------------
def translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))
