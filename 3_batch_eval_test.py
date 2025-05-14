import os
import tensorflow as tf
import numpy as np
from PIL import Image
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

# Nacitanie modelu a mien tried
model = tf.keras.models.load_model("keras_model.h5")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Funkcia na klasifikaciu jedneho obrazku
def classify_image(image_path):
    # Nacitanie obrazka, prevod na RGB a zmena velkosti na 224x224
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    # Normalizujeme hodnoty pixelov
    image_array = np.array(image) / 255.0
    # Pridame batch dimenziu, model ocakava vstup vo forme batchu
    image_array = np.expand_dims(image_array, axis=0)
    # Predikcia modelu, vratene pravdepodobnosti pre kazdu triedu
    predictions = model.predict(image_array)
    # Ziskame index triedy s najvyssou pravdepodobnostou
    predicted_index = np.argmax(predictions)
    # Vratime nazov triedy a pravdepodobnost predikcie
    return class_names[predicted_index], float(predictions[0][predicted_index])

# Cesty
test_folder = "test_images"  # Zlozka s testovacimi obrazkami
output_csv = "klasifikacne_vysledky.csv"  # Vystupny CSV subor s vysledkami

results = []  # Zoznam na ukladanie vysledkov klasifikacie
per_class_stats = defaultdict(lambda: {"correct": 0, "total": 0})  # Statistika pre kazdu triedu

# Prejdeme cez vsetky obrazky v zlozke test_images
for filename in os.listdir(test_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(test_folder, filename)

        # Zo nazvu suboru extrahujeme spravny nazov triedy
        true_label = filename.split("_")[0].lower()

        # Klasifikujeme obrazok a ziskame predikovanu triedu a pravdepodobnost
        predicted_label, confidence = classify_image(image_path)
        # Vyberieme predikovany nazov triedy
        predicted_class_name = predicted_label.split(" ", 1)[1].lower()
        # Porovname spravnu triedu a predikovanu triedu
        is_correct = true_label == predicted_class_name

        # Ulozime vysledky do zoznamu
        results.append([filename, true_label, predicted_class_name, confidence, "✓" if is_correct else "✗"])

        # Aktualizujeme statistiku pre tuto triedu
        per_class_stats[true_label]["total"] += 1
        if is_correct:
            per_class_stats[true_label]["correct"] += 1

# Ulozenie vysledkov do CSV suboru
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Obrazok", "Spravna trieda", "Predikcia", "Presnost (%)", "Spravne"])
    for row in results:
        writer.writerow([row[0], row[1], row[2], f"{row[3]*100:.2f}", row[4]])

print("\n📊 Presnost po triedach:")
for class_label, stats in per_class_stats.items():
    # Vypocitame presnost pre kazdu triedu
    accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    print(f" - {class_label}: {accuracy:.2f}% ({stats['correct']} / {stats['total']})")

# Vypocet priemernej presnosti napriec vsetkymi triedami
total_accuracy = 0
accuracies = {}
num_classes = len(per_class_stats)

for class_label, stats in per_class_stats.items():
    # Vypocitame presnost pre kazdu triedu
    class_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    total_accuracy += class_accuracy
    accuracies[class_label] = class_accuracy

# Vypocitame priemernu presnost
average_accuracy = total_accuracy / num_classes if num_classes > 0 else 0
print(f"\n📈 Priemerna presnost napriec triedami: {average_accuracy:.2f}%")

print(f"\n✅ Vysledky ulozene v subore: {output_csv}")

# Vygenerovanie grafu presnosti pre jednotlive triedy
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylim(0, 110)
plt.ylabel("Presnost (%)")
plt.xlabel("Trieda")
plt.title("Presnost klasifikacie po triedach")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("graf_presnosti.png")
plt.show()