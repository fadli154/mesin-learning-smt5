# Perbandingan Lengkap Baseline, Model Sebelum Tuning, dan Best Estimator — Pertemuan 12 (`PERBANDINGAN_MODEL_P12.md`)

**Mata Kuliah:** Pembelajaran Mesin (*Machine Learning*) — Pertemuan 12  
**Mahasiswa:** Fadli Hifziansyah (`241730042` | Kelas `5 INF B`)  
**Sumber Eksperimen Terkunci:** `Pertemuan 12/241730042_Fadli_Hifziansyah_P12_TugasML_FINAL.ipynb`  
**Dataset & Partisi:** UCI #697 (`dataset-asli.csv`, $N = 4.424$; `SEED = 42` $\rightarrow$ *Train* $n = 2.654$, *Validation* $n = 885$, *Test* $n = 885$; `28` fitur mentah konservatif $\rightarrow$ `213` kolom hasil transformasi array *dense* pada *Training Set*)

---

## 1. Spesifikasi Tiga Kelompok Model yang Dibandingkan

| Kelompok Pembanding | Varian yang Diuji | Spesifikasi Teknis di dalam `Pipeline` | Jumlah Fit Pelatihan |
|---|---|---|---:|
| **1. Baseline Tanpa Pembelajaran Pola** | `DummyClassifier (prior)` | `ColumnTransformer` + `DummyClassifier(strategy='prior', random_state=42)` | `5` CV + `1` fit penuh |
| **2. Model Sebelum Tuning (Acuan P11)** | **2a.** *Native* (`C = 1,0`)<br>**2b.** `+ Isotonic CV=5` *(Final P11)* | `ColumnTransformer` + `LogisticRegression(C=1.0, class_weight=None, l1_ratio=0.0, solver='lbfgs', max_iter=2000, random_state=42)` | `5` CV + `1` fit penuh<br>(+ `5` fit kalibrasi untuk 2b) |
| **3. Best Estimator Hasil Tuning P12** | **3a.** *Native* (`C = 0,3`) *[Utama P12]*<br>**3b.** `+ Isotonic CV=5` | `ColumnTransformer` + `LogisticRegression(C=0.3, class_weight=None, l1_ratio=0.0, solver='lbfgs', max_iter=2000, random_state=42)` terpilih dari `GridSearchCV` (`refit='AP'`) | `60` CV fit + `1` `refit` = `61` fit<br>(+ `5` fit kalibrasi untuk 3b) |

---

## 2. Perbandingan pada 5-Fold Stratified Cross-Validation (*Training Set*, $n = 2.654$)

*Catatan: Seluruh simpangan baku (`SD`) dihitung seragam menggunakan standar deviasi populasi `np.std(..., ddof=0)`.*

| Model / Kandidat | Mean Train AP | Mean CV AP ± SD (`ddof=0`) | Gap (Train − CV AP) | Mean CV ROC-AUC ± SD (`ddof=0`) | Norma Koefisien `\|\|w\|\|_2` (Refit Penuh) |
|---|---:|---:|---:|---:|---:|
| **1. Baseline `DummyClassifier (prior)`** | `0,3214` | `0,3214 ± 0,0008` | `0,0000` | `0,5000 ± 0,0000` | — |
| **2. Sebelum Tuning (`C = 1,0, class_weight = None`)** | `0,8530` | `0,8114 ± 0,0137` | `+0,0416` | `0,8732 ± 0,0135` | `6,4629` (`76` iterasi) |
| **3. Best Estimator (`C = 0,3, class_weight = None`)** | `0,8456` | **`0,8131 ± 0,0136`** | **`+0,0324`** | **`0,8756 ± 0,0137`** | `4,3923` (`-32,0%`; `64` iterasi) |
| *(Kontras Grid Terburuk: `C = 10,0, 'balanced'`)* | `0,8566` | `0,7954 ± 0,0201` | `+0,0612` | `0,8635 ± 0,0125` | *(Tidak di-refit)* |
| *(Kontras Grid Underfitting: `C = 0,03, None`)* | `0,8201` | `0,8018 ± 0,0134` | `+0,0182` | `0,8715 ± 0,0138` | *(Tidak di-refit)* |

---

## 3. Perbandingan pada *Validation Set* ($n = 885$) — Diskriminasi, Kalibrasi, & Pemilihan Threshold

Matriks biaya kesalahan menggunakan satuan biaya relatif simulasi (`TN = 0`, `FP = 1`, `FN = 4`, `TP = 0`) dengan skenario simulasi kapasitas tindak lanjut `<= 35%` (`<= 309` mahasiswa di *Validation Set*).

### A. Evaluasi pada Ambang Default `t = 0,50` (*Validation Set*)
| Model | Val ROC-AUC | Val AP | Val Brier | Val ECE-10 | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | Flagged (%) | Biaya (1:4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Baseline `DummyClassifier (prior)`** | `0,5000` | `0,3209` | `0,2179` | `0,0005` | `0` | `0` | `601` | `284` | `0,0000` | `0,0000` | `1,0000` | `0,0000` | `0 (0,00%)` | `1.136` |
| **2a. Sebelum Tuning — Native (`C = 1,0`)** | `0,8852` | `0,8393` | `0,1142` | `0,0444` | `201` | `46` | `555` | `83` | `0,8138` | `0,7077` | `0,9235` | `0,7571` | `247 (27,91%)` | `378` |
| **3a. Best Estimator P12 — Native (`C = 0,3`)** | `0,8847` | `0,8396` | `0,1140` | `0,0346` | `198` | `45` | `556` | `86` | `0,8148` | `0,6972` | `0,9251` | `0,7514` | `243 (27,46%)` | `389` |
| **2b. Sebelum Tuning + Isotonic (Final P11)** | **`0,8868`** | **`0,8408`** | **`0,1136`** | **`0,0270`** | `197` | `46` | `555` | `87` | `0,8107` | `0,6937` | `0,9235` | `0,7476` | `243 (27,46%)` | `394` |
| **3b. Best Estimator P12 + Isotonic (`C = 0,3`)** | `0,8835` | `0,8364` | `0,1148` | `0,0310` | `203` | `51` | `550` | `81` | `0,7992` | `0,7148` | `0,9151` | `0,7546` | `254 (28,70%)` | **`375`** |

### B. Evaluasi pada Ambang Operasional Terkunci (`<= 35%` Kapasitas di *Validation Set*)
| Model | `t*` Eksak Terkunci | Alternatif Grid `0,01` | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | Flagged (%) | Biaya Val (1:4) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Baseline `DummyClassifier (prior)`** | `0,5000` | — | `0` | `0` | `601` | `284` | `0,0000` | `0,0000` | `1,0000` | `0,0000` | `0 (0,00%)` | `1.136` |
| **2a. Sebelum Tuning — Native (`C = 1,0`)** | `0,3580` | `0,36` (`Cost=354`) | `219` | `90` | `511` | `65` | `0,7087` | `0,7711` | `0,8502` | `0,7386` | `309 (34,92%)` | `350` |
| **3a. Best Estimator P12 — Native (`C = 0,3`)** | **`0,3515`** | `0,35` (`Cost=345`) | `220` | `89` | `512` | `64` | `0,7120` | `0,7746` | `0,8519` | `0,7420` | `309 (34,92%)` | `345` |
| **2b. Sebelum Tuning + Isotonic (Final P11)** | `0,3483` | `0,34` (`Cost=344`) | `220` | `86` | `515` | `64` | `0,7190` | `0,7746` | `0,8569` | `0,7458` | `306 (34,58%)` | **`342`** |
| **3b. Best Estimator P12 + Isotonic (`C = 0,3`)** | `0,3794` | `0,37` (`Cost=355`) | `215` | `76` | `525` | `69` | `0,7388` | `0,7570` | `0,8735` | `0,7478` | `291 (32,88%)` | `352` |

---

## 4. Perbandingan Evaluasi Akhir Satu Kali pada *Test Set* ($n = 885$)

### A. Hasil *Test Set* pada Ambang Default `t = 0,50`
| Model | Test ROC-AUC | Test AP | Test Trap PR-AUC | Test Brier | Test ECE-10 | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | Biaya Test (1:4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Baseline `DummyClassifier (prior)`** | `0,5000` | `0,3209` | `0,6605` | `0,2179` | `0,0005` | `0` | `0` | `601` | `284` | `0,0000` | `0,0000` | `1,0000` | `0,0000` | `1.136` |
| **2a. Sebelum Tuning — Native (`C = 1,0`)** | `0,8913` | `0,8273` | `0,8269` | `0,1171` | `0,0280` | `183` | `47` | `554` | `101` | `0,7957` | `0,6444` | `0,9218` | `0,7121` | `451` |
| **3a. Best Estimator P12 — Native (`C = 0,3`)** | **`0,8929`** | **`0,8284`** | **`0,8280`** | **`0,1168`** | **`0,0209`** | `177` | `41` | `560` | `107` | `0,8119` | `0,6232` | `0,9318` | `0,7052` | `469` |
| **2b. Sebelum Tuning + Isotonic (Final P11)** | `0,8898` | `0,8217` | `0,8237` | `0,1184` | `0,0234` | `182` | `45` | `556` | `102` | `0,8018` | `0,6408` | `0,9251` | `0,7123` | `453` |
| **3b. Best Estimator P12 + Isotonic (`C = 0,3`)** | `0,8919` | `0,8229` | `0,8250` | `0,1176` | `0,0368` | `186` | `46` | `555` | `98` | `0,8017` | `0,6549` | `0,9235` | `0,7209` | **`438`** |

### B. Hasil *Test Set* pada Ambang Operasional Terkunci dari *Validation Set*
| Model | Threshold Val Terkunci | TP | FP | TN | FN | Precision | Recall | Specificity | F1 | Flagged (%) | Biaya Test (1:4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Baseline `DummyClassifier (prior)`** | `0,5000` | `0` | `0` | `601` | `284` | `0,0000` | `0,0000` | `1,0000` | `0,0000` | `0 (0,00%)` | `1.136` |
| **2a. Sebelum Tuning — Native (`C = 1,0`)** | `0,3580` | `212` | `76` | `525` | `72` | `0,7361` | `0,7465` | `0,8735` | **`0,7413`** | `288 (32,54%)` | **`364`** *(Terendah di Test)* |
| **3a. Best Estimator P12 — Native (`C = 0,3`)** | `0,3515` | `209` | `80` | `521` | `75` | `0,7232` | `0,7359` | `0,8669` | `0,7295` | `289 (32,66%)` | `380` |
| **2b. Sebelum Tuning + Isotonic (Final P11)** | `0,3483` | `207` | `76` | `525` | `77` | `0,7314` | `0,7289` | `0,8735` | `0,7302` | `283 (31,98%)` | `384` |
| **3b. Best Estimator P12 + Isotonic (`C = 0,3`)** | `0,3794` | `201` | `70` | `531` | `83` | `0,7417` | `0,7077` | `0,8835` | `0,7243` | `271 (30,62%)` | `402` |

---

## 5. Sintesis Kritis Tiga Dimensi Keputusan dan Justifikasi Artikel UAS

1. **Pemenang menurut `Mean CV AP` (*Best Estimator* Hasil Tuning):**
   `LogisticRegression(C = 0,3, class_weight = None, l1_ratio = 0.0)` menang tipis atas konfigurasi pra-*tuning* `C = 1,0` pada validasi silang (`Mean CV AP`: `+0,0017`; `Mean CV ROC-AUC`: `+0,0024`) serta konsisten meraih nilai terbaik pada seluruh metrik peringkat dan kalibrasi bebas ambang di *Test Set* (`Test AP = 0,8284`, `Test ROC-AUC = 0,8929`, `Test Brier = 0,1168`, `Test ECE = 0,0209`). Selisih ini **sangat kecil** (di bawah 1 SD antar-fold `0,0136`) dan tidak diklaim signifikan secara statistik.
2. **Perbedaan antara Peringkat `AP` Tertinggi dan Biaya Operasional Terendah:**
   Model dengan `AP` tertinggi **tidak otomatis** memberikan biaya keputusan biner terendah pada satu ambang spesifik. Pada *Validation Set*, biaya terendah diraih oleh `C = 1,0 + Isotonic` (`342` satuan), sedangkan pada *Test Set* biaya operasional terendah diraih oleh `C = 1,0` *native* (`364` satuan), diikuti `C = 0,3` *native* (`380` satuan) dan `C = 1,0 + Isotonic` (`384` satuan). Sesuai disiplin metodologi, pilihan hasil *tuning* **tidak diubah surut** setelah membaca angka *Test Set*.
3. **Paragraf Justifikasi Konfigurasi untuk Artikel Ilmiah UAS:**
   > Berdasarkan protokol *hyperparameter tuning* terkendali pada Pertemuan 12 yang melanjutkan Pertemuan 11, konfigurasi utama yang direkomendasikan untuk dilaporkan pada artikel ilmiah Proyek UAS adalah `Pipeline` prapemrosesan 28 fitur konservatif akhir Semester Pertama (`ColumnTransformer` yang memetakan 28 fitur mentah menjadi 213 kolom hasil transformasi pada *Training Set* sebagai array *dense*) dengan pengklasifikasi `LogisticRegression(C = 0,3, class_weight = None, l1_ratio = 0.0, solver = 'lbfgs', max_iter = 2000)` serta ambang operasional `t* = 0,3515` yang dikunci dari *Validation Set* (skenario biaya relatif `1:4` dan batas kapasitas simulasi `35%`), disertai pelaporan berdampingan konfigurasi pra-*tuning* `C = 1,0` (*native* `t* = 0,3580` dan *Isotonic* P11 `t* = 0,3483`) sebagai analisis pembanding yang transparan. Pemilihan `C = 0,3` dan `class_weight = None` dipertahankan secara disiplin mengikuti aturan pra-komitmen validasi silang (`Mean CV AP` tertinggi `0,8131 ± 0,0136`, `Mean CV ROC-AUC = 0,8756 ± 0,0137`) serta terbukti konsisten meraih peringkat dan kalibrasi *native* terbaik pada *Test Set* (`Test AP = 0,8284`, `Test ROC-AUC = 0,8929`, `Test Brier = 0,1168`, `Test ECE = 0,0209`; norma koefisien menyusut `32,0%` dari `6,4629` menjadi `4,3923` sebagai konsekuensi regularisasi $L_2$). Pada saat yang sama, naskah artikel wajib menyatakan secara jujur bahwa kenaikan `AP` terhadap `C = 1,0` sangat kecil (`+0,0017` di CV; `+0,0011` di *Test Set*) dan bahwa skor `AP` tertinggi tidak identik dengan biaya keputusan operasional terendah (`Test Cost` `C = 0,3` *native* sebesar `380` satuan berbanding `384` satuan pada P11 `C = 1,0 + Isotonic` dan `364` satuan pada `C = 1,0` *native*), sehingga kontribusi ilmiah artikel bertumpu pada ketatnya metodologi evaluasi bebas kebocoran, bukan klaim peningkatan akurasi yang dramatis.
