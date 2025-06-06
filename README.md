# 🧠 Klasifikácia objektov pomocou umelých neurónových sietí

Tento projekt obsahuje tri implementácie testovania trénovaného modelu na rozpoznávanie objektov. Model bol vytvorený pomocou Teachable Machine a exportovaný vo formáte Keras (h5). Testovanie je implementované v jazyku Python.

## 📁 Obsah repozitára

| Súbor/Priečinok              | Popis                                                                 |
|-----------------------------|-----------------------------------------------------------------------|
| `1_webcam_test.py`          | Spustenie reálneho testovania cez **webkameru** pomocou OpenCV.       |
| `2_gui_upload_test.py`      | **GUI aplikácia** (Tkinter), ktorá umožňuje nahrať obrázok a zobraziť predikciu. |
| `3_batch_eval_test.py`      | **Automatizované testovanie** obrázkov z priečinka `test_images/`. Výsledky sa ukladajú do CSV a generuje sa graf presnosti. |
| `keras_model.h5`            | Trénovaný model neurónovej siete.                                    |
| `labels.txt`                | Zoznam tried, ktoré model rozpoznáva.                               |
| `klasifikacne_vysledky.csv` | Výsledky automatizovaného testovania pre jednotlivé obrázky.              |
| `graf_presnosti.png`        | Grafická vizualizácia presnosti klasifikácie podľa tried.            |
| `test_images/`              | Priečinok s testovacími obrázkami.                                   |
