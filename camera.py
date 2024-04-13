import cv2
from tkinter import *
from PIL import Image, ImageTk

def show_frame():
    global is_live, frame_captured
    if is_live:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if ret:
            frame_captured = frame  # Save the last captured frame to use when capturing photo

            # Convert the image from BGR (OpenCV) to RGB
            cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_rgb)

            # Convert image for tkinter
            imgtk = ImageTk.PhotoImage(image=img)
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)

            # Repeat after an interval to capture continuously
            image_label.after(10, show_frame)

def capture_image():
    global is_live
    if frame_captured is not None:
        is_live = False  # Stop updating the live feed

        # Convert the last captured frame to RGB
        cv_image = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(cv_image)
        img = ImageTk.PhotoImage(image=im)

        # Display the captured photo instead of the live feed
        image_label.configure(image=img)
        image_label.image = img

def retake_photo():
    global is_live
    is_live = True  # Resume the live feed
    show_frame()

def close_camera():
    # Release the webcam and close the application window
    cap.release()
    root.destroy()

# Initialize the main window
root = Tk()
root.title("Webcam App")

# Initialize webcam
cap = cv2.VideoCapture(1)  

# State variables
is_live = True
frame_captured = None  # To store the last frame captured

# Create a label to display the images
image_label = Label(root)
image_label.pack(padx=5, pady=5)

# Buttons frame
buttons_frame = Frame(root)
buttons_frame.pack(padx=5, pady=5)

# Create a button to capture images
capture_button = Button(buttons_frame, text="Capture Photo", command=capture_image)
capture_button.pack(side=LEFT, padx=5)

# Create a button to retake the photo
retake_button = Button(buttons_frame, text="Retake Photo", command=retake_photo)
retake_button.pack(side=LEFT, padx=5)

# Create a button to close the application
close_button = Button(buttons_frame, text="Close", command=close_camera)
close_button.pack(side=LEFT, padx=5)

# Start showing the frame
show_frame()

# Start the GUI event loop
root.mainloop()
