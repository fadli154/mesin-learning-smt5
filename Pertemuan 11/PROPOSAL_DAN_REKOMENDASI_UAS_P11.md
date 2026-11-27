# Rekomendasi Threshold Operasional & Proposal Singkat Artikel UAS — Pertemuan 11

**Nama Mahasiswa:** Fadli Hifziansyah  
**NIM:** `241730042`  
**Kelas:** `5 INF B`  
**Mata Kuliah:** Pembelajaran Mesin (*Machine Learning*) — Pertemuan 11  
**Dosen Pengampu:** Bapak Ahmad Tabrani  
**Notebook Utama:** `241730042_Fadli_Hifziansyah_P11_TugasML.ipynb`

---

## Bagian A — Ringkasan Rekomendasi Threshold Operasional untuk Artikel UAS

1. **Model Klasifikasi Utama yang Dikunci:**  
   Pipeline `LogisticRegression(C=1.0)` dengan kalibrasi probabilitas eksplisit `CalibratedClassifierCV(method='isotonic', cv=5)` yang dilatih murni pada *Training Set* ($n = 2.654$) menggunakan **28 fitur spesifikasi konservatif akhir Semester Pertama** pada dataset *Predict Students' Dropout and Academic Success* (UCI Machine Learning Repository ID #697; DOI Dataset: `10.24432/C5MC89`).
2. **Alasan Pentingnya Kalibrasi Probabilitas:**  
   Pada *Validation Set* ($n = 885$, prevalensi positif `32,09%`), model dengan `class_weight='balanced'` tanpa kalibrasi mengalami distorsi estimasi peluang (`Brier Score = 0,1306`, `ECE 10-Bin = 0,0975`), sedangkan **`LogReg + Isotonic (CV=5)`** mencapai kinerja diskriminasi **`ROC-AUC = 0,8868`** dan **`Average Precision (AP) = 0,8408`** sekaligus kualitas kalibrasi terbaik (**`Brier Score = 0,1136`**, **`ECE 10-Bin = 0,0270`**).
3. **Pertimbangan Biaya Kesalahan, Prevalensi, dan Skenario Simulasi Kapasitas Acuan:**  
   * Penalti kesalahan dinormalisasi ke satuan biaya relatif dengan **$C_{\text{FP}} = 1$ satuan** dan skenario acuan asimetris **$C_{\text{FN}} = 4$ satuan** (diuji sensitivitasnya pada rasio $1:1$, $1:2$, $1:4$, dan $1:8$).
   * Tanpa kendala kapasitas, biaya minimum pada *Validation Set* berada di **`t = 0,26`** (`313 satuan` pada grid `0,01`; eksak `t = 0,2682` dengan `312 satuan`), namun menandai **`42,03%` kohort** (`372` mahasiswa), melampaui skenario simulasi kapasitas acuan **`<= 35%` kohort** (`<= 309` mahasiswa; *bukan kapasitas empiris institusi*).
   * Berdasarkan pencarian eksak pada seluruh `488` titik probabilitas unik dengan kendala kapasitas `<= 35%`, biaya minimum eksak tercapai pada **`t* = 0,3483`** (`Biaya = 342 satuan`, `Flagged = 306` / `34,58%`, `TP = 220`, `FP = 86`, `FN = 64`), sedangkan pembulatan operasional pada grid `0,01` menghasilkan **`t_final = 0,34`** (`Biaya = 344 satuan`, `Flagged = 308` / `34,80%`, `TP = 220`, `FP = 88`, `FN = 64`).
4. **Rekomendasi yang Dikunci & Hasil Evaluasi Akhir Satu Kali pada *Test Set* ($n = 885$):**  
   * Ambang operasional yang direkomendasikan untuk artikel UAS adalah **`t_final = 0,34`** (dengan pelaporan pendamping titik eksak `0,3483`).
   * Pada *Unseen Test Set* ($n = 885$), konfigurasi terkunci menghasilkan **`ROC-AUC = 0,8898`**, **`AP = 0,8217`**, **`Brier Score = 0,1184`**, **`ECE 10-Bin = 0,0234`**, **`Precision = 72,82%`**, **`Recall = 73,59%`**, **`Specificity = 87,02%`**, **`F1-Score = 0,7320`**, menandai **`287` mahasiswa (`32,43%`)**, serta memangkas biaya kesalahan sebesar **`75 satuan` (`-16,56%`)** dan mendeteksi **`27` mahasiswa *dropout* lebih banyak** dibandingkan ambang pembanding baku `t = 0,50` (`453 satuan`).

---

## Bagian B — Proposal Singkat Artikel Ilmiah Proyek UAS (Target SINTA 4)

### 1. Judul Kerja Artikel
**Optimasi Ambang Keputusan Sensitif Biaya dan Kalibrasi Probabilitas pada Model Regresi Logistik untuk Prediksi Risiko Putus Studi Mahasiswa Pasca-Semester Pertama**

### 2. Latar Belakang & Rumusan Masalah
Banyak penelitian klasifikasi risiko putus studi (*student dropout*) pada pendidikan tinggi hanya mengevaluasi akurasi atau `ROC-AUC` pada ambang keputusan bawaan `0,50`, mengikutsertakan variabel akademik semester akhir yang berisiko menimbulkan *temporal data leakage*, serta mengabaikan apakah skor probabilitas model telah terkalibrasi ketika digunakan dalam pengambilan keputusan berbasis biaya asimetris dan keterbatasan kuota layanan bimbingan akademik.

### 3. Pertanyaan Penelitian (*Research Questions*)
* **RQ1:** Bagaimana kinerja diskriminasi (`ROC-AUC`, *Average Precision*, dan *Trapezoidal PR-AUC*) model Regresi Logistik dibandingkan *baseline* `DummyClassifier` dan model pembanding `DecisionTreeClassifier` dalam memprediksi risiko *dropout* menggunakan 28 fitur spesifikasi konservatif pada akhir Semester Pertama?
* **RQ2:** Bagaimana dampak kalibrasi probabilitas eksplisit (*Platt/Sigmoid Scaling* dan *Isotonic Regression*) dibandingkan probabilitas bawaan dan `class_weight='balanced'` terhadap *Brier Score* serta *Expected Calibration Error* (ECE)?
* **RQ3:** Bagaimana pemilihan *threshold* operasional berdasarkan matriks biaya kesalahan asimetris ($1:1$ s.d. $1:8$) dan skenario simulasi kapasitas tindak lanjut memengaruhi *Precision*, *Recall*, *Specificity*, *F1-Score*, serta generalisasi biaya pada *Test Set*?

### 4. Dataset dan Pengendalian Validitas Fitur
* **Dataset:** *Predict Students' Dropout and Academic Success* ($N = 4.424$ mahasiswa, 36 fitur prediktor + 1 kolom `Target`: `Graduate = 2.209`, `Dropout = 1.421`, `Enrolled = 794`), *Instituto Politécnico de Portalegre*, Portugal.
* **Sitasi Resmi:** Repositori UCI ID #697 (`DOI: 10.24432/C5MC89`, Lisensi `CC BY 4.0`); artikel deskripsi dataset oleh Realinho et al. (2022, `DOI: 10.3390/data7110146`).
* **Definisi Target & Waktu Prediksi:** Klasifikasi biner `Dropout = 1` (`32,12%`) vs. status tidak dilabeli `Dropout` pada *endpoint* pengamatan `Non-Dropout = 0` (`67,88%`), dilengkapi analisis sensitivitas eksklusi `Enrolled` ($N = 3.630$) untuk menguji efek *right-censoring*. Titik prediksi dikunci pada akhir Semester Pertama dengan mengeluarkan 6 fitur Semester Kedua serta membandingkan spesifikasi konservatif 28 fitur terhadap spesifikasi pembanding 30 fitur (`Debtor` & `Tuition fees up to date`).

### 5. Baseline, Model Utama, Model Pembanding, dan Metrik Evaluasi
* **Baseline:** `DummyClassifier(strategy='prior')`, `DummyClassifier(strategy='stratified')`, serta ambang acuan bawaan `t = 0,50`.
* **Model Utama:** Pipeline `ColumnTransformer` + `LogisticRegression(C=1.0)` + `CalibratedClassifierCV(method='isotonic', cv=5)`.
* **Model Pembanding:** `LogisticRegression` (*Native* & *Balanced*), `LogisticRegression + Platt/Sigmoid`, serta `DecisionTreeClassifier` (*Pruned* `max_depth=5, min_samples_leaf=20` & *Unpruned*).
* **Metrik Evaluasi:** Diskriminasi (`ROC-AUC`, `AP`, `Trapezoidal PR-AUC`), Kalibrasi (`Brier Score`, `ECE` 10-bin & 5-bin, *Reliability Diagram*), Klasifikasi per *Threshold* (`TP, FP, TN, FN, Accuracy, Precision, Recall, Specificity, F1`), serta `Total Error Cost` lintas skenario rasio biaya dan kapasitas.

### 6. Rencana Tahapan Eksperimen & Calon Jurnal Target SINTA 4
* **Tahapan Eksperimen:**
  1. *Tahap 1 (Pertemuan 11 — Selesai):* Audit kualitas data & fitur, pembagian berstrata `60/20/20` dengan *Test Set* terisolasi, kalibrasi probabilitas, evaluasi tiga *Confusion Matrix* (`0,30`, `0,50`, `0,70`), pencarian *threshold* eksak & grid berbasis biaya-kapasitas, *Error Analysis*, dan evaluasi akhir *Test Set*.
  2. *Tahap 2 (Pertemuan 12–13):* Analisis interpretabilitas koefisien fitur / *permutation importance* dan perluasan pembanding ansambel pohon pada partisi data yang sama.
  3. *Tahap 3 (Pertemuan 14–15):* Penulisan naskah lengkap format IMRaD (sitasi IEEE) sesuai pedoman jurnal target SINTA 4.
* **Calon Jurnal Target SINTA 4 (Bukti Awal di Folder `bukti_sinta/`):**
  1. **KOMPUTA : Jurnal Ilmiah Komputer dan Informatika (UNIKOM)** — `SINTA ID: 8075` (SINTA 4; OJS: https://ojs.unikom.ac.id/index.php/komputa).
  2. **Infotek : Jurnal Informatika dan Teknologi (Universitas Hamzanwadi)** — `SINTA ID: 7721` (SINTA 4; OJS: https://e-journal.hamzanwadi.ac.id/index.php/infotek).
  3. **JIKO (Jurnal Informatika dan Komputer) — UTDI Yogyakarta** — `SINTA ID: 103` (SINTA 4; OJS: https://ejournal.utdi.ac.id/index.php/jiko).
