# Laporan Audit Akhir Pertemuan 12 (`AUDIT_FINAL_P12.md`)

**Mata Kuliah:** Pembelajaran Mesin (*Machine Learning*) — Pertemuan 12  
**Mahasiswa:** Fadli Hifziansyah (`241730042` | Kelas `5 INF B`)  
**Dosen Pengampu:** Bapak Ahmad Tabrani  
**Berkas Notebook Final:** `Pertemuan 12/241730042_Fadli_Hifziansyah_P12_TugasML.ipynb`  
**Sumber Dataset Lokal:** `Pertemuan 12/dataset-asli.csv` (`533.230` byte; UCI #697, SHA-256 LF: `19737f02127e85ef29af536e46888a7721b6c530c3e1c6aedfbb09109c030a71`)

---

## 1. Daftar Koreksi Dokumentasi & Metodologi Akhir

Seluruh rancangan eksperimen, `Pipeline`, `GridSearchCV` 12 kandidat $\times$ 5 fold (`61` total `fit`), dan angka hasil evaluasi dipertahankan tanpa perubahan eksperimen maupun pengeditan angka manual. Empat penyempurnaan ketepatan ilmiah yang diterapkan meliputi:

1. **Penjelasan Akurat Penggunaan Label pada Stratifikasi vs. Isolasi *Test Set* (Bagian B.4 & C.3–C.5):**
   * Klaim harfiah bahwa *Test Set* *"sama sekali tidak dimasukkan maupun dibaca"* telah diganti dengan penjelasan teknis yang presisi: kolom `Target` digunakan pada tahap awal **hanya** untuk pemetaan kelas biner dan stratifikasi pembagian data (`train_test_split(..., stratify=df_uci["Target"])`) agar proporsi kelas konsisten. Setelah partisi terbentuk, fitur dan label *Test Set* tidak digunakan dalam proses *fitting*, *cross-validation*, pemilihan *hyperparameter*, kalibrasi, maupun penentuan *threshold*, dan baru dievaluasi pada tahap akhir (Bagian G) setelah seluruh konfigurasi dikunci. Keterbatasan bahwa partisi *Test Set* yang sama pernah dievaluasi pada tahap akhir Pertemuan 11 tetap dipertahankan secara jujur.
2. **Peran `random_state` pada `LogisticRegression` dengan `solver='lbfgs'` (Bagian B.1 & C.3):**
   * Dijelaskan secara eksplisit sesuai dokumentasi `Scikit-learn 1.9.1` bahwa parameter `random_state` pada `LogisticRegression` hanya digunakan oleh *solver* berbasis pengacakan (`sag`, `saga`, `liblinear`) dan **tidak memengaruhi optimasi deterministik `solver='lbfgs'`**. Reproduksibilitas eksperimen dikendalikan oleh `SEED = 42` pada `train_test_split` dan `StratifiedKFold`, sementara `l1_ratio=0.0` tetap ditetapkan eksplisit untuk regularisasi murni $L_2$.
3. **Pelunakan Klaim *Underfitting* pada `C = 0,03` (Bagian E.1, E.6, & H.2 `log_tuning`):**
   * Mengingat kesenjangan *train–CV* pada `C = 0,03` relatif kecil (`Mean Train AP = 0,8201`; `Mean CV AP = 0,8018 ± 0,0134`; `gap = +0,0182`), istilah mutlak *underfitting* telah dilunakkan menjadi **indikasi regularisasi terlalu kuat atau *underfitting* relatif terhadap kandidat terbaik (`C = 0,3`)**.
4. **Penjelasan Objektif `class_weight = None` terhadap Kalibrasi (Bagian E.1 & E.6):**
   * Klaim bahwa `class_weight = None` otomatis menghasilkan probabilitas lebih terkalibrasi telah diubah menjadi penjelasan teoretis bahwa opsi tersebut **tidak melakukan pembobotan ulang antarkelas**, sedangkan kualitas kalibrasinya dibuktikan secara empiris melalui `Brier Score` dan `ECE` aktual pada Bagian F.

*(Seluruh koreksi sebelumnya—`28` fitur mentah $\rightarrow$ `213` kolom array *dense*, penyusutan norma koefisien `6,4629` $\rightarrow$ `4,3923` [`32,0%`], keseragaman `np.std(..., ddof=0)`, pemisahan `Best CV AP` [`380` satuan] vs. biaya uji terendah `C = 1,0` *native* [`364` satuan], serta Cleveland Dot Plot dengan catatan *zoomed range*—tetap terjaga utuh).*

---

## 2. Status *Clean Execution* (*Restart Kernel and Run All*)

* **Status Ketersediaan Lingkungan & Dataset:** Terverifikasi tersedia (`Python 3.14.5`, `NumPy 2.4.4`, `Pandas 3.0.6`, `Scikit-learn 1.9.1`; `Sumber dataset aktif: Berkas lokal (dataset-asli.csv)`).
* **Urutan Eksekusi Bersih:** Seluruh *output* lama dikosongkan dan seluruh `15` sel kode dieksekusi berurutan dari kernel baru dengan **`execution_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`** dan **`0` *error***.
* **Waktu Terukur Terkini (`Cell [6]` / Bagian E.4):** `wall-clock GridSearchCV = 4,9819 detik` (`2026-11-28 20:39:55.001` s.d. `20:39:59.983`), akumulasi *fitting* 60 fold `5,9951 detik`, `Mean ± SD [ddof=0]` waktu 1x `fit` `0,0999 ± 0,0216 detik`, `refit_time_ = 0,0642 detik`.
* **Verifikasi Otomatis (`Cell [15]` / Bagian H.8):** Baik pemeriksaan invarian rumus dinamis (termasuk panjang kesimpulan `373` kata dalam rentang `250–400` kata) maupun pemeriksaan regresi spesifik eksperimen lulus **100%**.
