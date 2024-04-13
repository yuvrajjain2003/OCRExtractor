import pytesseract
import PIL.Image
import cv2
import numpy as np
import re

# img = cv2.imread("test_image.jpeg")
# cv2.imshow("Original Image", img)
# cv2.waitKey(0)


def main():
    extract_text("test_image.jpeg")

def preprocess_image(image_path):
    """Preprocess the image."""

    # Read the image
    img = cv2.imread("test_image.jpeg")

    # Grayscale image
    gray_image = grayscale(img)

    # Threshold image
    thresh, bw_image = cv2.threshold(gray_image, 55, 65, cv2.THRESH_BINARY)

    # Remove Noise
    no_noise = noise_removal(bw_image)

    # Thicken the text (Dilation)
    dilated_image = thick_font(no_noise)

    # TODO: Deskew the image. (Currently not working)
    
    return dilated_image


def extract_text(image_path):
    """Extract the text and retrieve Serial Number and Model Number."""
    preprocessed_image = preprocess_image(image_path)

    # PSM = Page Segmentation Mode
    # OEM = OCR Engine Mode
    config = r"--psm 6 --oem 3"
    text = pytesseract.image_to_string(preprocessed_image, config=config)

    print("THE TEXT READ FROM IMAGE: \n")
    print(text)

    # Regex pattern to find the Model number
    model_pattern = r"Model (\w+)"
    # Regex pattern to find the Serial number
    serial_pattern = r"Serial (\w+)"

    # Search the text for the model and serial numbers
    model_match = re.search(model_pattern, text)
    serial_match = re.search(serial_pattern, text)

    # Extract the model and serial number if found
    model_number = model_match.group(1) if model_match else None
    serial_number = serial_match.group(1) if serial_match else None

    print("EXTRACTED MODEL NUMBER AND SERIAL NUMBER: \n")
    print(model_number, serial_number)
    
    return (model_number, serial_number)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

# Source: (https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df)
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


def rotateImage(cvImage, angle: float):
    """Rotate the image around its center."""
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    """Deskew image."""
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


if __name__ == "__main__":
    main()