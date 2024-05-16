import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')
# Check if the cascade file has been loaded correctly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

clock_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_wallclock.xml')  
# Check if the clock cascade file has been loaded correctly
if clock_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

eye_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_eye.xml')
# Check if the eye cascade file has been loaded correctly
if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

# Check age and gender
ageProto = "Age/deploy_age.prototxt"
ageModel = "Age/age_net.caffemodel"

genderProto = "Gender/deploy_gender.prototxt"
genderModel = "Gender/gender_net.caffemodel"

ageNet=cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Function to detect faces in live camera stream
def detect_faces():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Define the scaling factor
    scaling_factor = 1
    # Iterate until the user hits the 'Esc' key
    while True:
        # Capture the current frame
        _, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, None, 
                fx=scaling_factor, fy=scaling_factor, 
                interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the face
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

        # Display the number of faces detected
        num_faces = len(face_rects)
        text = f'Number of faces: {num_faces}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the output
        cv2.imshow('Face Detector', frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Release the video capture object
    cap.release()

    # Close all the windows
    cv2.destroyAllWindows()

# Function to detect faces in live camera stream
def detect_eyes():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)
    # Define the scaling factor
    ds_factor = 1

    # Iterate until the user hits the 'Esc' key
    while True:
        # Capture the current frame
        _, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # For each face that's detected, run the eye detector
        for (x,y,w,h) in faces:
            # Extract the grayscale face ROI
            roi_gray = gray[y:y+h, x:x+w]

            # Extract the color face ROI
            roi_color = frame[y:y+h, x:x+w]

            # Run the eye detector on the grayscale ROI
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Draw circles around the eyes
            for (x_eye,y_eye,w_eye,h_eye) in eyes:
                # Draw square around the eyes
                cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 255, 0), 2)
                
                # Add text "eyes" above the square
                cv2.putText(roi_color, 'Eye', (x_eye, y_eye - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the output
        cv2.imshow('Eye Detector', frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Release the video capture object
    cap.release()

    # Close all the windows
    cv2.destroyAllWindows()

# Function to detect age and gender in live camera stream
def detect_age_and_gender():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Define the scaling factor
    scaling_factor = 1

    # Iterate until the user hits the 'Esc' key
    while True:
        # Capture the current frame
        ret, frame = cap.read()

        # Check if frame is captured correctly
        if not ret:
            print("Error: Unable to capture video frame.")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, None, 
                fx=scaling_factor, fy=scaling_factor, 
                interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around the face
        for (x,y,w,h) in face_rects:

            # Extract the face region
            face = frame[y:y+h, x:x+w]
            
            # Prepare the blob for age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]
            gender_confidence_score = genderPred[0][genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]
            age_confidence_score = agePred[0][agePred[0].argmax()]

            label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
            # label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
            print(label)
            yPos = y - 15
            while yPos < 15:
                yPos += 15
            box_color = (0,255,0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (x, y), (x+w,y+h), box_color, 3)
            # Label processed image
            font_scale = 0.54
            cv2.putText(frame, label, (x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

        # Display the output
        cv2.imshow('Face Detector', frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Release the video capture object
    cap.release()

    # Close all the windows
    cv2.destroyAllWindows()

# Function to detect faces in an uploaded image
def detect_faces_in_image(uploaded_image):
    # Convert the uploaded image file to a NumPy array
    img_array = np.array(Image.open(uploaded_image))
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
    )
    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

        num_faces = len(faces)
        text = f'Number of faces: {num_faces}'
        cv2.putText(img_array, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the resulting image with face detection
    return img_array

# Function to detect eyes in an uploaded image
def detect_eyes_in_image(uploaded_image):
    # Convert the uploaded image file to a NumPy array
    img_array = np.array(Image.open(uploaded_image))
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

     # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
    )

    # For each face that's detected, run the eye detector
    for (x,y,w,h) in faces:
        # Extract the grayscale face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Extract the color face ROI
        roi_color = img_array[y:y+h, x:x+w]  # Change 'frame' to 'img_array'

        # Run the eye detector on the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Draw squares around the eyes and add "eyes" text
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            # Draw square around the eyes
            cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 255, 0), 2)
            
            # Add text "eyes" above the square
            cv2.putText(roi_color, 'Eye', (x_eye, y_eye - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting image with face and eye detection
    return img_array

def detect_age_gender_in_image(uploaded_image):
    # Convert the uploaded image file to a NumPy array
    img_array = np.array(Image.open(uploaded_image))
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
    )
    # Draw a rectangle around each detected face and predict age and gender
    for (x, y, w, h) in faces:
        # Extract the face region
        face = img_array[y:y+h, x:x+w]

        # Prepare the blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        gender_confidence_score = genderPred[0][genderPred[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        age_confidence_score = agePred[0][agePred[0].argmax()]

        # Label with age and gender
        label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
        print(label)
        yPos = y - 15
        while yPos < 15:
            yPos += 15
        box_color = (0, 255, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(img_array, (x, y), (x+w, y+h), box_color, 3)
        font_scale = 0.54
        cv2.putText(img_array, label, (x, yPos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

    # Display the resulting image with face detection
    return img_array

def process_image(image_path, process_function):
    processed_image = process_function(image_path)
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(processed_image_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Tạo một cửa sổ mới để hiển thị hình ảnh
    new_window = tk.Toplevel(root)
    new_window.title("Processed Image")

    # Label để hiển thị hình ảnh trong cửa sổ mới
    label_img = tk.Label(new_window)
    label_img.pack()

    label_img.config(image=img_tk)
    label_img.image = img_tk

def upload_and_process(process_function):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path, process_function)

# Create main window
root = tk.Tk()
root.title("Ứng dụng nhận diện khuôn mặt")

# Create and place buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_detect_faces = tk.Button(frame_buttons, text="Nhận diện số người", command=lambda: detect_faces())
btn_detect_faces.pack()

btn_detect_age_gender = tk.Button(frame_buttons, text="Nhận diện tuổi và giới tính", command=lambda: detect_age_and_gender()) 
btn_detect_age_gender.pack()

btn_detect_eyes = tk.Button(frame_buttons, text="Nhận diện mắt", command=lambda: detect_eyes() )
btn_detect_eyes.pack()

# Add buttons for each functionality
btn1 = tk.Button(root, text="Detect Faces", command=lambda: upload_and_process(detect_faces_in_image))
btn1.pack()

btn2 = tk.Button(root, text="Detect Age and Gender", command=lambda: upload_and_process(detect_age_gender_in_image))
btn2.pack()

btn4 = tk.Button(root, text="Detect Eyes", command=lambda: upload_and_process(detect_eyes_in_image))
btn4.pack()

label_img = tk.Label(root)
label_img.pack()

# Start the GUI event loop
root.mainloop()