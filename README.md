# Belajar ANN (Algoritma Backpropagation)

Simulator interaktif dan kalkulator Multi-Layer Perceptron (MLP) dengan algoritma Backpropagation untuk mata kuliah Pembelajaran Mesin.

## Fitur Utama

- **Latihan & Kalkulator Interaktif:** Menghitung 2 iterasi penuh (Forward & Backward Propagation, total 46 parameter/gradien).
- **Kustomisasi Parameter Dinamis:** Bobot awal, bias, dataset input masukan, dan learning rate dapat diubah langsung pada tabel yang menyerupai antarmuka IKN.
- **Tipografi Matematika Native:** Notasi pecahan parsial dan simbol matematika ter-render tajam dan instan tanpa ketergantungan jaringan offline.
- **Infografis Step-by-Step & Galeri Lengkap:** Dilengkapi 5 tahap visualisasi proses komputasi interaktif.

## Arsitektur Jaringan

- **Input Layer:** 2 neuron ($X_1, X_2$)
- **Hidden Layer:** 1 neuron ($A_1$) dengan fungsi aktivasi Sigmoid
- **Output Layer:** 1 neuron ($\hat{Y}$) dengan fungsi aktivasi Sigmoid
- **Fungsi Loss:** $J = \frac{1}{2}(\hat{Y} - Y)^2$
