from camera import capture_image
from extractor import extract_text, analyze_text

def main():
    camera_index = 1  # Adjust this index based on your earlier testing
    clear_text_obtained = False

    while not clear_text_obtained:
        image = capture_image(camera_index)
        if image is not None:
            text = extract_text(image)
            clear_text_obtained = analyze_text(text)

if __name__ == "__main__":
    main()