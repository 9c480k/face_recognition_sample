#functions 
#need it to scan and log a face, then recognise it the next time around 

import face_recognition
import numpy as np
import os, json 
from sklearn.svm import SVC
import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2

def build_encodings(dataset_path): #can consider trying to store it in a json file instead or use a db structure for faster searching 
    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
            else:
                raise Exception(f"Warning: No face found in {img_path}")

    return known_names, known_encodings


def store_encodings(known_names, known_encodings):
    #need to convert from nparray to list 
    serialised_encodings = []
    for encoding in known_encodings: 
        serialised_encodings.append(encoding.tolist())

    input_list = [known_names, serialised_encodings]

    with open('encodings.json', 'w') as file:
        json.dump(input_list, file, indent=4)


def retrieve_encodings():
    try: 
        with open("encodings.json", "r") as file:
            data = json.load(file)
    except: 
        known_names, known_encodings = build_encodings(r"C:\Users\lucas\Documents\stored_faces")
        store_encodings(known_names, known_encodings)
        
        with open("encodings.json", "r") as file:
            data = json.load(file)

    unserialised_list = []
    for encoding in data[1]: 
        unserialised_list.append(np.array(encoding))

    return data[0], unserialised_list


def validation(dataset_path):
    test_encodings = []
    file_names = []
    #quick and dirty way of building encodings for all the validation images 
    for img_file in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            test_encodings.append(encodings[0])
            file_names.append(img_file)
        else:
            raise Exception(f"Warning: No face found in {img_path}")
        
    #should probably include some comparison algo here to get the general percentage instead of just returning the embeddings 
    #working with a small validation set tho so too lazy to code that in 

    return file_names, test_encodings



#run this part if you want to refresh the encodings stored already 
#known_names, known_encodings = build_encodings(r"C:\Users\lucas\Documents\stored_faces")
#store_encodings(known_names, known_encodings)

known_names, known_encodings = retrieve_encodings()
#note: using a classifier makes the model unable to give you an "unknown" result 
#it will always give you the most likely result, regardless of how low the probability is 
#can try optimising it for a certain probability threshold, but better to use face recognition's inbuilt compare faces function

#clf = SVC()
#clf.fit(known_encodings, known_names)

#test_names, test_encodings = validation(r"C:\Users\lucas\Documents\test_images")
#print(test_names, [clf.predict([encoding])[0] for encoding in test_encodings])


           

#name = clf.predict(test_encoding)
#print(name)
#name = clf.predict([face_encoding])[0]


NUM_IMAGES = 10

def capture_images():
    person_name = simpledialog.askstring("Input", "Enter the person's name:")

    #makes sure there is a person tagged to the image
    if not person_name:
        return

    dataset_dir = os.path.join(r"C:\Users\lucas\Documents\stored_faces", person_name)
    try: 
        os.makedirs(dataset_dir, exist_ok=False)
    
    except: 
        #the directory already exists, no need to make dirs 
        pass 
 

    #might want to handle this better to make sure that there arent duplicate directories of the same person 
    

    cap = cv2.VideoCapture(0)
    count = 0
    messagebox.showinfo("Info", f"capturing {NUM_IMAGES} images, press Q to cancel early.")

    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capturing Faces", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        img_path = os.path.join(dataset_dir, f"{person_name}_{count + 1}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

    cap.release()
    cv2.destroyAllWindows()   
    messagebox.showinfo("Currently saving. Please wait for the task to be completed.")     
    known_names, known_encodings = build_encodings(r"C:\Users\lucas\Documents\stored_faces")
    store_encodings(known_names, known_encodings)
    known_names, known_encodings = retrieve_encodings()
    messagebox.showinfo("Done", f"{count} images saved to {dataset_dir}")



    
def recognise_faces():
    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame from the webcam
        ret, frame = video_capture.read()

        # Resize frame to 1/4 size for faster processings
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(rgb_small_frame , cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if face matches known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Scale back up face locations since we scaled the frame down
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)

        # Display the result
        cv2.imshow('Webcam Face Recognition', frame)

        # Break with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close window
    video_capture.release()
    cv2.destroyAllWindows()


# --- GUI Setup ---
root = tk.Tk()
root.title("face verification tool thingy")
root.geometry("300x200")

label = tk.Label(root, text="face recogniser thing", font=("Arial", 14))
label.pack(pady=20)

btn_capture = tk.Button(root, text="register new face", command=capture_images)
btn_capture.pack(pady=10)
btn_recognise = tk.Button(root, text="recogniser", command=recognise_faces)
btn_recognise.pack(pady=10)

root.mainloop()


# Load known faces and their names
#known_face_encodings = []
#known_face_names = []

# Load a sample picture and learn how to recognize it
#person1_image = face_recognition.load_image_file("person1.jpg")
#person1_encoding = face_recognition.face_encodings(person1_image)[0]
#known_face_encodings.append(person1_encoding)
#known_face_names.append("Person 1")

# Add more people as needed
# person2_image = face_recognition.load_image_file("person2.jpg")
# person2_encoding = face_recognition.face_encodings(person2_image)[0]
# known_face_encodings.append(person2_encoding)
# known_face_names.append("Person 2")




'''
from sklearn.svm import SVC

encodings_array = []
names_array = []

with open('data.json') as json_data:
        nodes = json.load(json_data)
        for node in nodes:
            encodings_array.append(node['encoding'])
            names_array.append(node['name'])

'''



