import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Nacitanie modelu a mien tried
model = tf.keras.models.load_model("keras_model.h5")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Funkcia na klasifikaciu jedneho obrazka
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, predictions

# Funkcia na zobrazenie grafu
def show_prediction_chart(predictions, class_names):
    plt.figure(figsize=(6, 4))
    plt.barh(class_names, predictions[0], color='skyblue')
    plt.xlabel("Pravdepodobnosť")
    plt.title("Výstup modelu")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

# Hlavna trieda aplikacie
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikacia obrazkov")
        self.root.geometry("500x600")

        self.label_result = tk.Label(root, text="Nacitaj obrazok", font=("Arial", 16))
        self.label_result.pack(pady=10)

        self.canvas = tk.Canvas(root, width=224, height=224)
        self.canvas.pack(pady=10)

        self.btn_upload = tk.Button(root, text="Nacitat obrazok", command=self.load_image)
        self.btn_upload.pack(pady=5)

        self.btn_classify = tk.Button(root, text="Klasifikovat", command=self.classify)
        self.btn_classify.pack(pady=5)

        self.image_path = None
        self.tk_image = None
    
    # Funkcia na nahravanie obrazku
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazky", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).convert("RGB").resize((224, 224))
            self.tk_image = ImageTk.PhotoImage(image)
            self.canvas.create_image(112, 112, image=self.tk_image)

    def classify(self):
        if self.image_path:
            result, predictions = classify_image(self.image_path)
            self.label_result.config(text=f"Vysledok: {result}")
            show_prediction_chart(predictions, class_names)
        else:
            self.label_result.config(text="Najprv nacitaj obrazok")

# Spustenie aplikacie
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
