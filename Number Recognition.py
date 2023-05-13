import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        self.canvas = tk.Canvas(self.master, width=300, height=300, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.button_clear = tk.Button(self.master, text="Clear", command=self.clear_all)
        self.button_clear.pack(side=tk.LEFT)
        self.button_predict = tk.Button(self.master, text="Predict", command=self.predict_digit)
        self.button_predict.pack(side=tk.RIGHT)
        self.image = Image.new("L", (300, 300), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label = tk.Label(self.master, text="", font=("Helvetica", 48))
        self.prediction_label.pack()

    def clear_all(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="")

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='white')

    def predict_digit(self):
        # Resize the image to 28x28 pixels and convert to grayscale
        img = self.image.resize((28, 28)).convert('L')

        # Normalize the image
        img = np.array(img) / 255.0

        # Reshape the image to support our model input
        img = img.reshape(1, 28, 28, 1)

        # Predict the class
        res = model.predict(img)[0]
        digit = np.argmax(res)

        # Update the prediction label
        self.prediction_label.config(text=str(digit))

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
