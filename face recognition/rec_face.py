import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from tkinter import *

# Face recognition code
# Looks through the face dataset folder and encodes all the faces to a format the face recognition can understand
# Image files can only be .jpg or .png
# Directory is /face dataset/
# Returns encoded faces
def encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./face dataset"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("face dataset/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

# Encodes the image names into a format the machine can understand
# Directory is /face dataset/
def encoded_image(img):

    face = fr.load_image_file("face dataset/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

# Classifies the face it was given image
# Finds all the faces in the image then labels them if the face is recognized from the face dataset
# If face isn't recognized it will be set to unrecognized
# Creates a list of faces
# Turns all the faces to keys
# Gets all face locations within the image
def classify_face(im):

    faces = encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(im, 1)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:

        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unrecognized"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Creates a box around the face
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

            # Creates a label and names the face
            cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_ITALIC
            cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting faces
    # Returns both faces that were recognized and unrecognized
    while True:

        cv2.imshow('face_rec', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names
        break
        cv2.destroyAllWindows()



# Classifies the file called recognize.jpg
def recognize():
    print(classify_face("recognize.jpg"))

# Creates a tkinter window titled 'window' with a black background
window = Tk()
window.title("window")
window.configure(background="black")

# Breaks the tkinter window into a top frame and bottom frame
topFrame = LabelFrame(window, background='black')
topFrame.pack()
bottomFrame = Frame(window, background='black')
bottomFrame.pack(side=BOTTOM)

# Text with program name information
name_text = "Project Name: Face Recognition" \
            "\nBy: Samael Newgate" \
            "\nClass: CSC44 - Deep Learning - SU19158" \
            "\nAssignment: 06"

# Text with program description text
description_text = "The program uses python and a face recognition library. \
                          \nIt will use face images within a dataset to classify the face of a selected image.\
                          \nThe GUI is made through tkinter "

# Text with program guide text
guide_text = "How to run:" \
             "\nStep 1: Go through all the tabs and learn about the program" \
             "\n\nStep 2: Press the Run button(green) to start the program" \
             "" \
             "\n\n\nHow to recognize your own pictures:" \
             "\nStep 1: Go to the face recognition folder > face dataset folder > \ndrag as many pictures of faces as you want that are png and jpg format" \
             "\n\nStep 2: Make sure to name each face with the name you want associated with that face" \
             "\n\nStep 3: Go back to the main face recognition folder and drag a picture of a face you want to be recognize, \nmake sure it's a png or jpg image" \
             "\n\nStep 4: Rename the picture to 'recognize' " \
             "\n\nStep 5/ Run the program"

# Window that shows  name information text
def name_window():
    toplevel = Toplevel()
    label1 = Label(toplevel, text=name_text, height=0, width=100)
    label1.pack()

#Window that shows description text
def description_window():
    toplevel = Toplevel()
    label2 = Label(toplevel, text=description_text, height=0, width=100)
    label2.pack()

#Window that shows guide text,
def guide_window():
    toplevel = Toplevel()
    label3 = Label(toplevel, text=guide_text, height=0, width=100)
    label3.pack()

# Name button set to the top frame, name window is bound to this button
button1 = Button(topFrame, text='Name', padx=5, pady=5, command = name_window)
button1.pack(side=LEFT)

# Description button set to the top frame, description window is bound to this button
button2 = Button(topFrame, text='Description', padx=5, pady=5, command = description_window)
button2.pack(side=LEFT)

# Guide button set to the top frame, guide window is bound to this button
button3 = Button(topFrame, text='Guide', padx=5, pady=5, command = guide_window)
button3.pack(side=LEFT)

# Run button set to the top frame, recognize function is bound to this button
button4 = Button(topFrame, text='Run', background='green', padx=5, pady=5, command=recognize)
button4.pack(side=LEFT)

# Exit button set to the top frame, window.destroy is bound to this button which destroys all active windows
button5 = Button(topFrame, text='Exit', background='red', padx=5, pady=5,command=window.destroy)
button5.pack(side=LEFT)

# Photo for the menu is named photo1 and the file is set to menu_image
# Image is bound to the tkinter window
# Background is set to black
photo1 = PhotoImage(file="menu_image.png")
menuimage = Label(window, image=photo1, background='black')
menuimage.pack()

# Main loop for the tkinter window
window.mainloop()


