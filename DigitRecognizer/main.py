import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab
from keras.models import load_model


def clear_all():
    global cv
    cv.delete("all")


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black',
                   capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def Recognize_Digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = cv

    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    # read image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply Otsu thresholding
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # get bounding box and extract ROI (region of interest)
        x, y, w, h = cv2.boundingRect(cnt)
        # create rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        # extract the image roi
        roi = th[y - top: y + h + bottom, x - left: x + w + right]
        # resize roi image to 28x28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # reshaping the image to support our model input
        img = img.reshape(1, 28, 28, 1)
        # normalizing the image to support our model input
        img = img/255.0
        # predicting result
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' (' + str(int(max(pred)*100)) + '%)'
        # cv2.putText() method is used to draw a text string on image.
        font, fontScale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        cv2.putText(image, data, (x, y-5), font, fontScale, color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey(0)


model = load_model(r'model.h5')
print("Model loaded successfully, continue in App")

# create the main window
root = Tk()
root.resizable(0, 0)
root.title("Digit Recognition App")

lastx, lasty = None, None
image_number = 0

# create canvas
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)


cv.bind('<Button-1>', activate_event)
btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear", command=clear_all)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
