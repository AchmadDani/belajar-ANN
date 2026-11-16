# Perhitungan Manual Artificial Neural Network (ANN) - Algoritma Backpropagation
**Mata Kuliah:** Pembelajaran Mesin / Machine Learning  
**Topik:** Multi-Layer Perceptron (Backpropagation 2 Iterasi)  
**Format Angka:** 5 Digit Desimal  

---

## 📋 Tabel Jawaban Soal

### Iterasi 1

#### Forward Propagation
| $Z_1$ | $A_1$ | $Z_2$ | $\hat{Y}$ |
|:---:|:---:|:---:|:---:|
| **1.22830** | **0.77352** | **0.65659** | **0.65849** |

#### Backward Propagation

**Baris 1: Turunan Parsial Aktivasi & Loss**
| $\frac{\partial J}{\partial A_2}$ | $\frac{\partial A_2}{\partial Z_2}$ | $\frac{\partial A_1}{\partial Z_1}$ |
|:---:|:---:|:---:|
| **0.65849** | **0.22488** | **0.17519** |

**Baris 2: Turunan Parsial Linear Combination**
| $\frac{\partial Z_2}{\partial W_2}$ | $\frac{\partial Z_2}{\partial b_2}$ | $\frac{\partial Z_2}{\partial A_1}$ | $\frac{\partial Z_1}{\partial W_{11}}$ | $\frac{\partial Z_1}{\partial W_{12}}$ | $\frac{\partial Z_1}{\partial b_1}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.77352** | **1.00000** | **0.28000** | **0.77000** | **0.18000** | **1.00000** |

**Baris 3: Gradien Loss terhadap Bobot & Bias**
| $\frac{\partial J}{\partial W_2}$ | $\frac{\partial J}{\partial b_2}$ | $\frac{\partial J}{\partial W_{11}}$ | $\frac{\partial J}{\partial W_{12}}$ | $\frac{\partial J}{\partial b_1}$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.11454** | **0.14808** | **0.00559** | **0.00131** | **0.00726** |

**Baris 4: Nilai Bobot & Bias Baru Terupdate**
| $W_2$ | $b_2$ | $W_{11}$ | $W_{12}$ | $b_1$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.25136** *(0.25137)* | **0.40298** | **0.28860** | **0.24967** | **0.95818** |

---

### Iterasi 2

#### Forward Propagation
| $Z_1$ | $A_1$ | $Z_2$ | $\hat{Y}$ |
|:---:|:---:|:---:|:---:|
| **1.17207** | **0.76352** | **0.59490** *(0.59491)* | **0.64449** |

#### Backward Propagation

**Baris 1: Turunan Parsial Aktivasi & Loss**
| $\frac{\partial J}{\partial A_2}$ | $\frac{\partial A_2}{\partial Z_2}$ | $\frac{\partial A_1}{\partial Z_1}$ |
|:---:|:---:|:---:|
| **-0.35551** | **0.22912** | **0.18056** |

**Baris 2: Turunan Parsial Linear Combination**
| $\frac{\partial Z_2}{\partial W_2}$ | $\frac{\partial Z_2}{\partial b_2}$ | $\frac{\partial Z_2}{\partial A_1}$ | $\frac{\partial Z_1}{\partial W_{11}}$ | $\frac{\partial Z_1}{\partial W_{12}}$ | $\frac{\partial Z_1}{\partial b_1}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.76352** | **1.00000** | **0.25136** *(0.25137)* | **0.62000** | **0.14000** | **1.00000** |

**Baris 3: Gradien Loss terhadap Bobot & Bias**
| $\frac{\partial J}{\partial W_2}$ | $\frac{\partial J}{\partial b_2}$ | $\frac{\partial J}{\partial W_{11}}$ | $\frac{\partial J}{\partial W_{12}}$ | $\frac{\partial J}{\partial b_1}$ |
|:---:|:---:|:---:|:---:|:---:|
| **-0.06219** | **-0.08146** | **-0.00229** | **-0.00052** | **-0.00370** |

**Baris 4: Nilai Bobot & Bias Baru Terupdate**
| $W_2$ | $b_2$ | $W_{11}$ | $W_{12}$ | $b_1$ |
|:---:|:---:|:---:|:---:|:---:|
| **0.26691** *(0.26692)* | **0.42334** | **0.28917** | **0.24980** | **0.95911** *(0.95910)* |

---

## 📖 Penjelasan Langkah Perhitungan

### 1. Informasi & Parameter yang Diketahui

- **Bobot (Weights) Awal:**
  - $W_{11} = 0.29000$
  - $W_{12} = 0.25000$
  - $W_2 = 0.28000$
- **Bias Awal:**
  - $b_1 = 0.96000$
  - $b_2 = 0.44000$
- **Learning Rate ($\alpha$):**
  - $\alpha = 0.25$
- **Dataset Latihan:**
  - **Data-1 (digunakan pada Iterasi 1):** $X_1 = 0.77000$, $X_2 = 0.18000$, target $Y = 0$
  - **Data-2 (digunakan pada Iterasi 2):** $X_1 = 0.62000$, $X_2 = 0.14000$, target $Y = 1$
- **Fungsi Aktivasi:** Sigmoid $\sigma(Z) = \frac{1}{1 + e^{-Z}}$ dengan turunan $\sigma'(Z) = \sigma(Z)(1 - \sigma(Z))$.
- **Loss Function:** $J = \frac{1}{2}(\hat{Y} - Y)^2$.

---

### 2. Perhitungan Terperinci: Iterasi 1

#### A. Forward Propagation
1. **Menghitung input linear hidden neuron ($Z_1$):**
   $$Z_1 = W_{11}X_1 + W_{12}X_2 + b_1$$
   $$Z_1 = (0.29000 \times 0.77000) + (0.25000 \times 0.18000) + 0.96000$$
   $$Z_1 = 0.22330 + 0.04500 + 0.96000 = \mathbf{1.22830}$$

2. **Menghitung aktivasi hidden neuron ($A_1$):**
   $$A_1 = \sigma(Z_1) = \frac{1}{1 + e^{-1.22830}} = \frac{1}{1 + 0.292790} = \mathbf{0.77352}$$

3. **Menghitung input linear output neuron ($Z_2$):**
   $$Z_2 = W_2 A_1 + b_2$$
   $$Z_2 = (0.28000 \times 0.77352) + 0.44000 = 0.216586 + 0.44000 = \mathbf{0.65659}$$

4. **Menghitung output prediksi network ($\hat{Y} = A_2$):**
   $$\hat{Y} = A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-0.65659}} = \frac{1}{1 + 0.518620} = \mathbf{0.65849}$$

---

#### B. Backward Propagation (Aturan Rantai / Chain Rule)

1. **Baris 1: Turunan Parsial Fungsi Aktivasi & Loss**
   - $\frac{\partial J}{\partial A_2} = A_2 - Y = 0.65849 - 0 = \mathbf{0.65849}$
   - $\frac{\partial A_2}{\partial Z_2} = A_2(1 - A_2) = 0.65849 \times (1 - 0.65849) = 0.65849 \times 0.34151 = \mathbf{0.22488}$
   - $\frac{\partial A_1}{\partial Z_1} = A_1(1 - A_1) = 0.77352 \times (1 - 0.77352) = 0.77352 \times 0.22648 = \mathbf{0.17519}$

2. **Baris 2: Turunan Parsial terhadap Linear Combination**
   - $\frac{\partial Z_2}{\partial W_2} = A_1 = \mathbf{0.77352}$
   - $\frac{\partial Z_2}{\partial b_2} = 1 = \mathbf{1.00000}$
   - $\frac{\partial Z_2}{\partial A_1} = W_2 = \mathbf{0.28000}$
   - $\frac{\partial Z_1}{\partial W_{11}} = X_1 = \mathbf{0.77000}$
   - $\frac{\partial Z_1}{\partial W_{12}} = X_2 = \mathbf{0.18000}$
   - $\frac{\partial Z_1}{\partial b_1} = 1 = \mathbf{1.00000}$

3. **Baris 3: Menghitung Gradien $\frac{\partial J}{\partial \theta}$**
   - **Gradien $W_2$:**
     $$\frac{\partial J}{\partial W_2} = \frac{\partial J}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial W_2} = 0.65849 \times 0.22488 \times 0.77352 = \mathbf{0.11454}$$
   - **Gradien $b_2$:**
     $$\frac{\partial J}{\partial b_2} = \frac{\partial J}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial b_2} = 0.65849 \times 0.22488 \times 1.00000 = \mathbf{0.14808}$$
   - **Gradien $W_{11}$:**
     $$\frac{\partial J}{\partial W_{11}} = \frac{\partial J}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_{11}}$$
     $$= 0.65849 \times 0.22488 \times 0.28000 \times 0.17519 \times 0.77000 = \mathbf{0.00559}$$
   - **Gradien $W_{12}$:**
     $$\frac{\partial J}{\partial W_{12}} = \frac{\partial J}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_{12}}$$
     $$= 0.65849 \times 0.22488 \times 0.28000 \times 0.17519 \times 0.18000 = \mathbf{0.00131}$$
   - **Gradien $b_1$:**
     $$\frac{\partial J}{\partial b_1} = \frac{\partial J}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial b_1}$$
     $$= 0.65849 \times 0.22488 \times 0.28000 \times 0.17519 \times 1.00000 = \mathbf{0.00726}$$

4. **Baris 4: Pembaruan Parameter (Gradient Descent Update): $\theta_{new} = \theta - \alpha \frac{\partial J}{\partial \theta}$**
   - $W_2^{(new)} = 0.28000 - (0.25 \times 0.11454) = 0.28000 - 0.028635 = \mathbf{0.25136}$ *(0.25137 jika pembulatan 5 desimal langsung)*
   - $b_2^{(new)} = 0.44000 - (0.25 \times 0.14808) = 0.44000 - 0.037020 = \mathbf{0.40298}$
   - $W_{11}^{(new)} = 0.29000 - (0.25 \times 0.00559) = 0.29000 - 0.001398 = \mathbf{0.28860}$
   - $W_{12}^{(new)} = 0.25000 - (0.25 \times 0.00131) = 0.25000 - 0.000328 = \mathbf{0.24967}$
   - $b_1^{(new)} = 0.96000 - (0.25 \times 0.00726) = 0.96000 - 0.001815 = \mathbf{0.95818}$

---

### 3. Perhitungan Terperinci: Iterasi 2

Pada Iterasi 2, input yang digunakan adalah **Data-2** ($X_1 = 0.62000, X_2 = 0.14000, Y = 1$) dan parameter menggunakan bobot/bias terupdate dari Iterasi 1:
- $W_{11} = 0.28860$
- $W_{12} = 0.24967$
- $W_2 = 0.25136$
- $b_1 = 0.95818$
- $b_2 = 0.40298$

#### A. Forward Propagation
1. **$Z_1$:**
   $$Z_1 = (0.28860 \times 0.62000) + (0.24967 \times 0.14000) + 0.95818 = 0.178932 + 0.034954 + 0.95818 = \mathbf{1.17207}$$
2. **$A_1$:**
   $$A_1 = \sigma(1.17207) = \frac{1}{1 + e^{-1.17207}} = \mathbf{0.76352}$$
3. **$Z_2$:**
   $$Z_2 = (0.25136 \times 0.76352) + 0.40298 = 0.191918 + 0.40298 = \mathbf{0.59490}$$
4. **$\hat{Y} = A_2$:**
   $$\hat{Y} = A_2 = \sigma(0.59490) = \frac{1}{1 + e^{-0.59490}} = \mathbf{0.64449}$$

---

#### B. Backward Propagation

1. **Baris 1:**
   - $\frac{\partial J}{\partial A_2} = A_2 - Y = 0.64449 - 1.00000 = \mathbf{-0.35551}$
   - $\frac{\partial A_2}{\partial Z_2} = A_2(1 - A_2) = 0.64449 \times (1 - 0.64449) = \mathbf{0.22912}$
   - $\frac{\partial A_1}{\partial Z_1} = A_1(1 - A_1) = 0.76352 \times (1 - 0.76352) = \mathbf{0.18056}$

2. **Baris 2:**
   - $\frac{\partial Z_2}{\partial W_2} = A_1 = \mathbf{0.76352}$
   - $\frac{\partial Z_2}{\partial b_2} = \mathbf{1.00000}$
   - $\frac{\partial Z_2}{\partial A_1} = W_2 = \mathbf{0.25136}$
   - $\frac{\partial Z_1}{\partial W_{11}} = X_1 = \mathbf{0.62000}$
   - $\frac{\partial Z_1}{\partial W_{12}} = X_2 = \mathbf{0.14000}$
   - $\frac{\partial Z_1}{\partial b_1} = \mathbf{1.00000}$

3. **Baris 3 (Gradien):**
   - $\frac{\partial J}{\partial W_2} = -0.35551 \times 0.22912 \times 0.76352 = \mathbf{-0.06219}$
   - $\frac{\partial J}{\partial b_2} = -0.35551 \times 0.22912 \times 1.00000 = \mathbf{-0.08146}$
   - $\frac{\partial J}{\partial W_{11}} = -0.35551 \times 0.22912 \times 0.25136 \times 0.18056 \times 0.62000 = \mathbf{-0.00229}$
   - $\frac{\partial J}{\partial W_{12}} = -0.35551 \times 0.22912 \times 0.25136 \times 0.18056 \times 0.14000 = \mathbf{-0.00052}$
   - $\frac{\partial J}{\partial b_1} = -0.35551 \times 0.22912 \times 0.25136 \times 0.18056 \times 1.00000 = \mathbf{-0.00370}$

4. **Baris 4 (Pembaruan Bobot & Bias):**
   - $W_2^{(new)} = 0.25136 - (0.25 \times -0.06219) = 0.25136 + 0.015548 = \mathbf{0.26691}$
   - $b_2^{(new)} = 0.40298 - (0.25 \times -0.08146) = 0.40298 + 0.020365 = \mathbf{0.42334}$
   - $W_{11}^{(new)} = 0.28860 - (0.25 \times -0.00229) = 0.28860 + 0.000572 = \mathbf{0.28917}$
   - $W_{12}^{(new)} = 0.24967 - (0.25 \times -0.00052) = 0.24967 + 0.000130 = \mathbf{0.24980}$
   - $b_1^{(new)} = 0.95818 - (0.25 \times -0.00370) = 0.95818 + 0.000925 = \mathbf{0.95911}$ *(0.95910 jika pembulatan step sebelumnya)*

