import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageDraw
import numpy as np
import os
import joblib
import torch
from cnn_model import CNNModel

# Constants
CANVAS_SIZE = 280
IMAGE_SIZE = 28
SAVE_DIR = "UserDrawings"
SCALER_PATH = "Scalers/mnist_scaler.pkl"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the StandardScaler
scaler = joblib.load(SCALER_PATH)

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a Digit")
        
        self.canvas = tk.Canvas(self.master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()
        
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)
        
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)

    def save_image(self):
        # Resize to 28x28 and invert colors
        resized_image = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS)
        img_array = np.array(resized_image)
        img_array = 255 - img_array  # Invert: white background to black background
        img_array = img_array.reshape(1, -1)  # Flatten for scaler

        # Normalize using StandardScaler
        img_array_scaled = scaler.transform(img_array)

        # Reshape back to 28x28
        img_array_scaled = img_array_scaled.reshape(IMAGE_SIZE, IMAGE_SIZE)

        # Prompt for filename
        filename = simpledialog.askstring("Save As", "Enter file name (without extension):")
        if filename:
            np.save(os.path.join(SAVE_DIR, f"{filename}.idx.npy"), img_array_scaled)
            print(f"Image saved as {SAVE_DIR}/{filename}.idx.npy")

def predict_drawing(file_path):
    # Load the image
    img_array = np.load(file_path)

    # Prepare for CNN
    cnn_input = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN model
    cnn_model = CNNModel().to(device)
    cnn_model.load_state_dict(torch.load("Models/cnn_best_model.pth", map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        output = cnn_model(cnn_input.to(device))
        prediction = torch.argmax(output, dim=1).item()
        print(f"CNN predicts: {prediction}")

    # Prepare for SVM
    svm_input = img_array.reshape(1, -1)
    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        svm_model = joblib.load(f"Models/svm_{kernel}.pkl")
        svm_prediction = svm_model.predict(svm_input)[0]
        print(f"SVM ({kernel}) predicts: {svm_prediction}")

def drawing_menu():
    while True:
        print("\nDrawing Menu:")
        print("1. Draw and Save Digit")
        print("2. Predict from Saved Drawing")
        print("0. Return to Main Menu")

        choice = input("Select an option: ")

        if choice == '1':
            root = tk.Tk()
            app = DrawingApp(root)
            root.mainloop()
        elif choice == '2':
            files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".npy")]
            if not files:
                print("No drawings found in UserDrawings/.")
                continue
            print("Available drawings:")
            for idx, file in enumerate(files):
                print(f"{idx + 1}. {file}")
            file_idx = int(input("Select file number: ")) - 1
            if 0 <= file_idx < len(files):
                predict_drawing(os.path.join(SAVE_DIR, files[file_idx]))
            else:
                print("Invalid selection.")
        elif choice == '0':
            break
        else:
            print("Invalid option. Please select again.")