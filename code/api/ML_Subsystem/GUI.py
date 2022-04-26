import time
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from ML_Manager import ML_Manager
from Paths_and_Keys import Paths_and_Keys


def open_image(image_row=2):
    global current_image, original_image

    filename = open_filename()
    original_image = Image.open(filename)
    current_image = original_image
    place_image()
    info_string.set('image uploaded.')

def place_image():
    global current_image, root, resize_factor, panel, info_string

    img = current_image
    img = img.resize((int(img.width/resize_factor), int(img.height/resize_factor)), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    # info_string.set('DONE!')


def open_filename():
    filename = filedialog.askopenfilename(title='"pen')
    return filename


def analyze():
    # Analyze Sample
    global current_image, manager, morphology_sequence, original_image, info_string

    morphology_sequence = input_text.get("1.0", "end-1c")

    final_img, analysis_text, pollens_dict = manager.analyze_sample(original_image, morphology_sequence=morphology_sequence)
    info_string.set('analyzed with morphology sequence: ' + morphology_sequence + '\n\nAnalysis Text: ' + analysis_text)
    current_image = final_img

    place_image()

def test():
    # Analyze Sample
    global current_image, manager, morphology_sequence, original_image, info_string

    morphology_sequence = input_text.get("1.0", "end-1c")

    current_image = manager.analyze_sample(original_image, morphology_sequence=morphology_sequence, test_extraction=True)
    info_string.set('extracted with morphology sequence: ' + morphology_sequence)

    place_image()


#######################################################################################################################
resize_factor = 3
manager = ML_Manager(load_model=True)

root = Tk()
root.title("Image Viewer")
root.geometry("1700x1200")

label_info = Label(root, text='PolliVidis Tester', font=("Arial", 25))
label_info.grid(row=0, columnspan=6)

# morphology
morphology_sequence = ''
label_morphology = Label(text="morphology sequence:")
input_text = Text(root, height=2, width=20, bg="light blue")

label_morphology.grid(row=2, column=0)
input_text.grid(row=2, column=1)

info_string = StringVar()
info_string.set("PolliVidis started...")
label_info = Label(root, textvariable=info_string)
label_info.grid(row=4, column=6)

# initial image
paths_and_keys = Paths_and_Keys()
original_image = Image.open(paths_and_keys.example_image_for_GUI_path)
current_image = original_image.resize((int(original_image.width / resize_factor), int(original_image.height / resize_factor)), Image.ANTIALIAS)
img = ImageTk.PhotoImage(current_image)
panel = Label(root, image=img)
panel.image = img
panel.grid(row=4, columnspan=5)

open_image_button = Button(root, text='open image', command=open_image, height=2, width=12)
open_image_button.grid(row=1, column=0)

button_test = Button(root, text="Extract", command=test, height=2, width=12)
button_analyze = Button(root, text="Analyze", command=analyze, height=2, width=12)
button_exit = Button(root, text="Exit", command=root.quit, height=2, width=12)

button_test.grid(row=2, column=2)
button_analyze.grid(row=2, column=3)
button_exit.grid(row=2, column=4)

root.resizable(width=True, height=True)
root.attributes('-fullscreen', True)
root.mainloop()
