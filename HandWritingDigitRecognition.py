from PIL import ImageGrab
import tensorflow as tf
import numpy as np
import tkinter as tk

# Load the models that you have already trained
model = tf.keras.models.load_model("./model/model.h5")
model.load_weights("./model/model_weights.h5")

# Create the tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Create the main canvas with black background color
canvas = tk.Canvas(root, width=280, height=250, bg='black')
canvas.pack()

screen = tk.Label(root, text="Draw a number", font=("Helvetica", 24))
screen.pack()


# Function to handle drawing on the canvas
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y


def draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    canvas.create_line((last_x, last_y, x, y), fill="white", width=10)
    last_x, last_y = x, y


# Function to predict the drawn digit
def predict_digit():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # There is a buggy area here
    # When I try to capture the canvas image sometimes it captures wrong places.
    # So I tried to fix it manually but, it still doesn't work perfect:(
    img = ImageGrab.grab((x+31, y+38, x1, y1))

    # See the captured image
    # img.show()

    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0  # Normalize input
    prediction = model.predict(img_array)

    # Hold the best prediction
    predicted_digit = np.argmax(prediction)
    screen.config(text="Predicted digit: " + str(predicted_digit))


# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    screen.config(text="Draw a number")


# Bind mouse events
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

reset_button = tk.Button(root, text="Clear", command=clear_canvas)
reset_button.pack()

# Start the main loop
root.mainloop()
