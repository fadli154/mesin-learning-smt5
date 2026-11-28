# Laporan Patch Final Audit Konsistensi & Validitas Pertemuan 12 (`AUDIT_FINAL_P12.md`)

**Mata Kuliah:** Pembelajaran Mesin (*Machine Learning*) — Pertemuan 12  
**Mahasiswa:** Fadli Hifziansyah (`241730042` | Kelas `5 INF B`)  
**Dosen Pengampu:** Bapak Ahmad Tabrani  
**Berkas Sebelumnya (Dipertahankan Utuh):** `Pertemuan 12/241730042_Fadli_Hifziansyah_P12_TugasML.ipynb`  
**Berkas Hasil Patch Final:** `Pertemuan 12/241730042_Fadli_Hifziansyah_P12_TugasML_FINAL.ipynb`  
**Berkas Perbandingan Lengkap Model:** `Pertemuan 12/PERBANDINGAN_MODEL_P12.md`  
**Dataset Lokal Disertakan:** `Pertemuan 12/dataset-asli.csv` (UCI #697, SHA-256 LF: `19737f02127e85ef29af536e46888a7721b6c530c3e1c6aedfbb09109c030a71`)

---

## 1. Ringkasan Audit dan Prinsip Perbaikan

Rancangan metodologis inti (`GridSearchCV` 12 kandidat $\times$ `5-Fold StratifiedKFold` $= 60$ CV fit $+ 1$ `refit` $= 61$ fit, `Pipeline` prapemrosesan `28` fitur konservatif akhir Semester Pertama, `refit="AP"`, `SEED = 42`, dan pembagian `60/20/20`) dipertahankan sepenuhnya tanpa perubahan eksperimen. Seluruh ketidaksesuaian antara keluaran kode aktual dan narasi dokumentasi telah diperbaiki secara tuntas pada `241730042_Fadli_Hifziansyah_P12_TugasML_FINAL.ipynb`.

---

## 2. Daftar Perubahan dan Angka yang Diperbaiki

| No | Aspek yang Diaudit | Kondisi Sebelum Patch | Koreksi pada `..._FINAL.ipynb` | Lokasi Bagian |
|:---:|---|---|---|---|
| **1** | **Jumlah kolom hasil transformasi** | Narasi menyebut `236` kolom (padahal `coef_.shape = (1, 213)` saat di-*fit* pada `X_train`, $n = 2.654$). | Seluruh penyebutan `236` dikoreksi menjadi **`213` kolom hasil transformasi** dan dibedakan tegas dari **`28` fitur mentah konservatif** sebelum preprocessing (diverifikasi dengan `assert clf_terbaik.coef_.shape == (1, 213)`). | A.4, B.1, D, E.1, E.7, G.4, H.4, H.8 |
| **2** | **Norma $L_2$ vektor koefisien (`\|\|w\|\|_2`)** | Narasi menyebut `5,5329` $\rightarrow$ `3,5297` (`36,2%`). | Dikoreksi sesuai *output* kode aktual menjadi **`6,4629` (`C = 1,0`) $\rightarrow$ `4,3923` (`C = 0,3`)**, menyusut **`32,0%`**, disertai penegasan bahwa penyusutan norma koefisien adalah konsekuensi matematis langsung regularisasi $L_2$, bukan bukti tunggal keunggulan model. | E.7, G.4, H.4, H.8 |
| **3** | **Istilah representasi *sparse* vs. *dense*** | Menyebut *"matriks fitur renggang (sparse one-hot)"* padahal `OneHotEncoder(..., sparse_output=False)`. | Diubah menjadi **"representasi *one-hot* dengan banyak nilai nol yang disimpan sebagai array *dense* (`sparse_output=False`)"**. | E.1, G.4 |
| **4** | **Referensi peta bagian A.2** | Menyebut `Bagian B.7` untuk penentuan metrik utama. | Dikoreksi menjadi **`Bagian B.4`** sesuai *heading* aktual (`### B.4 Pemilihan Metrik Utama...`) dan seluruh rujukan subbagian di A.2 diselaraskan. | A.2 |
| **5** | **Konsistensi simpangan baku (`SD`)** | `cv_results_` memakai SD populasi (`ddof=0`), sedangkan `cv_results_df["mean_fit_time"].std()` memakai SD sampel Pandas (`ddof=1`). | Seluruh perhitungan SD di Bagian D, E.4, dan H.2 disamakan secara eksplisit menggunakan **`np.std(..., ddof=0)`** (SD populasi atas lipatan/kandidat) dan diberi keterangan konvensi statistik. | D, E.4, H.2, H.4 |
| **6** | **Pemisahan *Best CV AP* vs. Biaya Operasional Minimum** | Penjelasan belum memisahkan cukup tajam bahwa pemenang `AP` bukan peraih biaya keputusan terendah. | Dipisahkan eksplisit menjadi tiga butir: (1) *Best estimator* menurut `Mean CV AP` (`C = 0,3, None`: `CV AP = 0,8131`, `Test AP = 0,8284`), (2) Konfigurasi berbiaya operasional *Test Set* terendah (`C = 1,0` *native* = **`364` satuan** < `C = 0,3` *native* = **`380` satuan** < P11 `C = 1,0 + Isotonic` = **`384` satuan**), dan (3) Justifikasi konfigurasi artikel UAS tanpa mengubah keputusan berdasarkan *Test Set*. | G.3, G.4, H.4 |
| **7** | **Klaim waktu `fit` kandidat `C = 10,0` & iterasi `lbfgs`** | Menyebut waktu `fit` `~2x` lebih lambat dan memastikan penyebabnya adalah jumlah iterasi `lbfgs`. | Rasio dihitung dinamis dari pengukuran aktual (`0,2736 / 0,2176 ≈ 1,26x` pada eksekusi kernel bersih final ini; sekitar `1,5x` pada eksekusi sebelumnya) dan ditegaskan bahwa karena `n_iter_` tidak direkam di seluruh 60 lipatan CV, penyebabnya tidak dipastikan berasal dari jumlah iterasi `lbfgs`. | E.1, E.6, H.2 |
| **8** | **Visualisasi peringkat 12 konfigurasi** | Grafik batang horizontal terpotong pada `0,76–0,84` memperbesar kesan visual selisih tipis. | Diganti menjadi **Cleveland Dot Plot dengan *Horizontal Error Bars* (`±1 SD`)** disertai kotak catatan eksplisit pada grafik bahwa sumbu `0,76–0,84` adalah *zoomed range* agar *error bar* terbaca (sedangkan *baseline* `Dummy` berada di `0,3214`). | E.6 (`01_evaluasi_gridsearch_cv.png`) |
| **9** | **Eksplisitasi `l1_ratio=0.0` pada `Scikit-learn 1.9.1`** | Mengandalkan nilai *default* `l1_ratio=0.0`. | Menetapkan **`l1_ratio=0.0` secara eksplisit** pada konstruktor `LogisticRegression(...)` dan mendokumentasikan bahwa menurut dokumentasi resmi `Scikit-learn 1.9.1`, parameter `penalty` telah *deprecated* sejak versi `1.8` dan dijadwalkan dihapus pada `1.10`. | B.1, C.3, H.8 |

---

## 3. Hasil Pengujian Eksekusi Ulang dari Kernel Bersih

1. **Ketersediaan Dataset & Lingkungan:**
   * Dataset lokal `Pertemuan 12/dataset-asli.csv` (`533.230` byte) terbaca langsung oleh kernel (`Sumber dataset aktif: Berkas lokal (dataset-asli.csv)`), terverifikasi SHA-256 penuh 64 karakter.
   * Lingkungan eksekusi: `Python 3.14.5`, `NumPy 2.4.4`, `Pandas 3.0.6`, `Scikit-learn 1.9.1`, `16` *logical CPU cores*.
2. **Audit Statis & Kebersihan Teks:**
   * `0` karakter kontrol terlarang, `0` simbol global tak terdefinisi, dan `0` kemunculan angka/istilah lama (`236`, `5,5329`, `3,5297`, `36,2%`, `Bagian B.7`, `sparse one-hot`, `~2x lebih lambat`).
3. **Eksekusi Berurutan Seluruh Sel (`nbclient`):**
   * Seluruh `31` sel (`16` Markdown + `15` sel kode) dieksekusi dari awal dalam kernel baru tanpa satu pun *error* (`0` *exception*).
   * Waktu pengukuran aktual eksekusi final: `wall-clock GridSearchCV = 8,6078 detik` (`2026-11-28 20:13:44.156` s.d. `20:13:52.764`), akumulasi *fitting* 60 fold `11,7044 detik` (+ *scoring* `1,7989 detik`), rata-rata 1x `fit` `0,1951 ± 0,0524 detik` (`ddof=0`), `refit_time_ = 0,1419 detik`.
4. **Pemeriksaan *Assertion* & Jumlah Kata Kesimpulan:**
   * Blok *Assertion* A (invarian metodologis & rumus dinamis) **LULUS 100%**, termasuk verifikasi panjang kesimpulan Bagian H.4 sebesar **`373` kata** (memenuhi rentang wajib `250–400` kata).
   * Blok *Assertion* B (regresi spesifik eksperimen UCI #697, `SEED = 42`) **LULUS 100%**, termasuk `coef_.shape == (1, 213)`, `l1_ratio == 0.0`, norma `6,4629 -> 4,3923 (32,0%)`, `best_params_ == {'clf__C': 0.3, 'clf__class_weight': None}`, `best_score_ == 0,8131`, dan biaya uji `364` vs. `380` vs. `384`.
5. **Perlindungan Berkas Sebelumnya:**
   * `04_TUGAS_P12_Hyperparameter_Tuning_yang_Terkendali.ipynb`, `241730042_Fadli_Hifziansyah_P12_TugasML.ipynb` (versi sebelum patch), seluruh notebook Pertemuan 11, dan `dataset-asli.csv` terverifikasi **100% identik byte-demi-byte**.
