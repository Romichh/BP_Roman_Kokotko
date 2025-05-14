import tensorflow as tf
import numpy as np
import cv2

# Nacitanie modelu a mien tried
model = tf.keras.models.load_model("keras_model.h5")
model.compile()  # Kompilacia (pre istotu)
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Spustenie kamery (0 znamena predna kamera / default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Priprava obrazku pre model (224x224, normalizacia)
    img = cv2.resize(frame, (224, 224))
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1  # Normalizacia na rozsah [-1, 1]

    # Predikcia modelom
    prediction = model.predict(img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    # Zobrazenie predikcie na obraz
    cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Zobrazenie obrazu s predikciou
    cv2.imshow("Klasifikacia objektov", frame)

    # Stlac 'q' pre ukoncenie
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnenie kamery a zatvorenie okien
cap.release()
cv2.destroyAllWindows()