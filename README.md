# Tubes2AI-Kelompok36

Repository ini berisi implementasi model deteksi serangan jaringan menggunakan **dataset UNSW-NB15**, yang berisi data paket jaringan mentah yang dihasilkan dengan menggunakan alat **IXIA PerfectStorm** oleh Cyber Range Lab UNSW Canberra. Dataset ini terdiri dari 10 jenis aktivitas, yang mencakup **9 jenis serangan** dan **1 aktivitas normal**. 

### Jenis Serangan dalam Dataset:
1. **Fuzzers**
2. **Analysis**
3. **Backdoors**
4. **DoS (Denial of Service)**
5. **Exploits**
6. **Generic**
7. **Reconnaissance**
8. **Shellcode**
9. **Worms**

Program yang dibuat diharapkan dapat memprediksi adanya serangan termasuk jenis serangan untuk suatu data paket jaringan.

## Cara Set up dan Run Program
- 1. Buka folder src, unduh file bertipe *Jupyter Notebook* (ekstensi .ipynb)
- 2. Jalankan seluruh *cell* pada file tersebut, pastikan seluruh proses download berhasil dan package tersedia untuk di-*import*
- 3. Perhatikan hasil *output* program, penjelasan lebih detail mengenai output tersebut terdapat pada file di folder doc

## Pembagian Tugas Anggota Kelompok

|  NIM   |    Nama Anggota       | Tugas      |
|--------|-----------------------|------------|
|10321009|Sahabista Arkitanego A.|  ID3, Data Balancer, Penjelasan singkat implementasi model (KNN, Naive-Bayes, ID3), Laporan pembentukan model ID3, Penjelasan proses Data Preprocessing, Pipeline.          |
|10821019|Dean Hartono           |    Penjelasan EDA, Set Up Program, Debugging        |
|12821046|Fardhan Indrayesa      | Feature Scaling, Feature Encoding, Dimensionality Reduction, KNN, Naive-Bayes, Confusion Matrix KNN, Confusion Matrix Naive-Bayes, laporan pembentukan model (KNN, Naive-Bayes), Perbandingan dengan referensi (KNN, Naive-Bayes), dan Evaluasi (KNN, Naive-Bayes           |
|13522051|Kharris Khisunica      |   EDA, Pipeline, Preprocessing         |
|18321011|Wikan Priambudi        |    Penjelasan EDA, Dealing with Outlier, Feature Engineering, ID3        |
