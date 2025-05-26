# Laporan Proyek Machine Learning - Jocelyn Nathaniel Patricktan 

## Domain Proyek

Rumah merupakan kebutuhan primer bagi manusia di mana setiap manusia membutuhkan rumah sebagai tempat untuk berlindung, hidup, berkembang, serta memberikan rasa aman dan nyaman. Utrecht merupakan kota dengan pertumbuhan populasi tercepat di Belanda [1]. Kota ini berkembang dari sebuah kota provinsi berukuran sedang menjadi ibukota regional yang dihuni oleh 350000 jiwa dan diperkirakan akan berkembang menjadi sekitar 455000 jiwa dalam 20 tahun ke depan yang berarti semakin banyak penduduk maka semakin banyak juga rumah yang dibutuhkan padahal lahan di kota tersebut sudah tidak banyak [1]. Tidak hanya itu, tetapi juga harga rumah menjadi sangat mahal bagi banyak penduduk, termasuk bagi generasi muda dan kelas menengah [1]. Pada tahun 2022, rata-rata harga rumah yang dihuni oleh pemilik rumah mencapai 511000 euro dengan harga jual naik 57% dalam empat tahun terakhir [2]. Krisis perumahan di kota Utrecht ini memang menjadi salah satu topik utama dalam pemilihan umum terakhir yang diadakan pada tanggal 22 November 2023 [1]. Permintaan akan perumahan sosial maupun perumahan yang terjangkau di kalangan penduduk semakin meningkat di mana waktu tunggu bagi penduduk untuk mendapatkan perumahan sosial telah mencapai sekitar 11 tahun selama bertahun-tahun serta penurunan kepemilikan perumahan yang terjangkau dari 19% menjadi 10% dan diperkirakan akan terus menurun di antara rentang tahun 2019 dan 2023 [1].  Tren permintaan rumah yang tinggi mengakibatkan semakin muda seseorang maka akan semakin kecil kemungkinan dalam mendapatkan perumahan sosial sehingga penduduk kaum muda cenderung memiliki beban sewa tertinggi terutama di pasar sewa pribadi [1]. Harga sewa rumah di kota Utrecht bahkan naik lebih cepat dibandingkan secara nasional di Belanda [2]. Dengan adanya krisis perumahan ini, tentunya penting bagi pembuat kebijakan dan perencana kota di kota Utrecht untuk mengetahui dan dapat memprediksi harga perumahan di mana prediksi ini akan digunakan untuk menentukan berapa harga jual yang pantas untuk perumahan dengan fitur tertentu ketika pemerintah ingin membangun lebih banyak perumahan sehingga penduduk kota Utrecht mampu untuk membeli rumah tersebut. 

[1] S. Aronica, “Housing rights: Utrecht’s challenges and solutions”, Energy Cities, 11 Maret 2025, [Online]. Tersedia: [Housing rights: Utrecht’s challenges and solutions - Energy Cities](https://energy-cities.eu/housing-rights-utrechts-challenges-and-solutions/) [Diakses: 25 Mei 2025]. 

[2] Utrecht Monitor, “Summary”, Gemeente Utrecht, 16 Mei 2023, [Online]. Tersedia: [Summary | Utrecht Monitor](https://utrecht-monitor.nl/samenvatting-utrecht-monitor/summary) [Diakses: 25 Mei 2025]. 


## Business Understanding

### Problem Statements

- Dari fitur-fitur yang terdapat pada dataset, fitur apa yang paling berpengaruh secara signifikan terhadap harga perumahan Utrecht?
- Bagaimana memprediksi harga perumahan Utrecht berdasarkan fitur-fitur tertentu? 

### Goals

- Mengetahui fitur-fitur apa saja yang paling berkorelasi dengan harga perumahan Utrecht. 
- Membuat model machine learning yang dapat digunakan untuk memprediksi harga perumahan Utrecht seakurat mungkin berdasarkan fitur-fitur yang tertera. 

### Solution Statement 

- Menggunakan dan membandingkan tiga algoritma regresi (karena harga merupakan variabel kontinu sehingga prediksi harga menggunakan regresi) seperti K-Nearest Neighbor (KNN), Random Forest, dan Boosting Algorithm untuk mengevaluasi model mana yang memiliki performa terbaik atau memiliki nilai kesalahan prediksi terkecil dalam memprediksi harga rumah di mana metrik evaluasi yang akan digunakan adalah Mean Squared Error (MSE). 

## Data Understanding
Pada proyek predictive analytics ini akan menggunakan Utrecht Housing Dataset Huge yang berisi informasi properti perumahan di kota Utrecht, Belanda. Dataset ini dikembangkan oleh Stefan Leijnen dan Sieuwert van Otterloo yang digunakan pada tiga hari pertama dari satu minggu kegiatan sekolah musim panas tentang “AI and machine learning” di Universitas Ilmu Terapan Utrecht pada bulan Juli 2022. Dataset ini tersedia secara public untuk keperluan riset dan edukasi bagi siapa pun. Dataset ini terdiri dari 2000 entri dan 16 fitur di mana masing-masing mewakili satu properti perumahan dengan berbagai macam fitur. 

Berikut merupakan tautan dataset yang diunduh dari Kaggle: [Utrecht housing dataset](https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset?select=utrechthousinghuge.csv)

### Variabel-variabel pada Utrecht Housing dataset adalah sebagai berikut: 

- id: merupakan angka antara 0 dan 100000 yang merupakan tanda pengenal unik untuk setiap rumah.
- zipcode: merupakan kode pos pada setiap rumah yang sesuai dengan area rumah tersebut.
- lot-len: merupakan panjang dalam meter dari sebidang tanah tempat rumah dibangun di mana setiap rumah dibangun di atas sebidang tanah persegi.
- lot-width: merupakan lebar dalam meter dari sebidang tanah tempat rumah dibangun di mana mulai dari 5.0 hingga 100.0 meter.
- lot-area: merupakan luas total dari kavling tanah tempat rumah dibangun di mana dapat dihitung dari lot-len dan lot-width.
- house-area: merupakan area tinggal rumah dalam meter persegi di mana 30.0 meter persegi merupakan rumah kecil.
- garden-size: merupakan ukuran taman dalam meter persegi.
- balcony: merupakan jumlah balkon yang dimiliki rumah tersebut di mana pada umumnya berupa 0, 1, atau 2 balkon.
- x-coor: merupakan koordinat x yang menggambarkan lokasi rumah yang berada di nilai bilangan bulat antara 2000 dan 3000.
- y-coor: merupakan koordinat y yang menggambarkan lokasi rumah yang berada di nilai bilangan bulat antara 5000 dan 6000.
- buildyear: merupakan tahun saat rumah dibangun di mana beberapa rumah tertua berasal dari tahun 1100 namun sebagian besar rumah dibangun pada abad ke-20.
- bathrooms: merupakan jumlah kamar mandi yang dimiliki rumah tersebut di mana sebagian besar rumah memiliki satu kamar mandi sementara beberapa rumah memiliki 2 atau 3 kamar mandi.
- taxvalue: merupakan nilai pajak rumah di mana angkanya berada di antara 50000 dan 1000000.
- retailvalue: merupakan nilai pasar sebuah rumah di mana angkanya berada di antara 50000 dan 1000000 dengan angka dibulatkan ke 1000 terdekat, variabel ini akan dijadikan sebagai fitur target.
- energy-eff: merupakan kehematan energi pada suatu rumah dengan nilai 1 berarti rumah tersebut hemat energi di mana hal tersebut penting untuk tujuan iklim tertentu.
- monument: merupakan nilai monumental di mana beberapa rumah di Utrecht terutama di rumah-rumah tua masih memiliki nilai tersebut karena memiliki desain arsitektur yang unik. 

## Data Preparation
Sebelum membangun model prediksi, diperlukan beberapa tahap penting pada data preparation dengan melakukan transformasi pada data untuk memastikan bahwa data yang digunakan bersih, konsisten, dan dalam bentuk yang sudah sesuai  atau cocok untuk proses pemodelan machine learning. Ada beberapa tahapan yang umum dilakukan pada data preparation seperti seleksi fitur, transformasi data, feature engineering, dan dimensionality reduction. Pada bagian ini, yang akan dilakukan adalah pembagian dataset dengan fungsi train_test_split dari library sklearn. Membagi data menjadi data latih (train) dan data uji (test) merupakan suatu hal yang harus dilakukan sebelum membuat pemodelan di mana perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Data uji (test set) berperan sebagai data baru sehingga perlu melakukan seluruh proses transformasi dalam data latih supaya tidak mengotori data uji dengan informasi yang didapat dari data latih. Maka dari itu, langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Proporsi pembagian data latih dan data uji yang akan digunakan pada proyek ini adalah 80:20. Hasil dari pembagian data latih dan data uji menunjukkan bahwa dari 1961 total jumlah sampel, akan diambil sebanyak 1568 untuk data latih dan sebanyak 393 untuk data uji. Selanjutnya untuk standarisasi, algoritma machine learning memiliki performa yang lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal, sehingga proses standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah untuk diolah oleh algoritma. Standarisasi adalah teknik transformasi yang paling umum digunakan sebagai tahap persiapan dalam pemodelan. Untuk fitur numerik, akan menggunakan teknik StandardScaler dari library Scikitlearn di mana akan melakukan proses standarisasi fitur dengan mengurangi mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi sehingga akan menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Standarisasi hanya akan dilakukan pada data latih lalu ketika sudah berada di tahap evaluasi baru akan dilakukan juga dengan data uji. Setelah dilakukan standarisasi, dapat terlihat bahwa nilai mean sudah = 0 dan standar deviasi = 1. 

![image](https://github.com/user-attachments/assets/7f876d43-7370-4eb7-a96a-ee787ef68f99)

## Modeling
Pada proyek ini, tiga algoritma regresi yang akan digunakan dan dibandingkan untuk memprediksi harga perumahan Utrecht, yaitu: 
1.	K-Nearest Neighbor (KNN)
   
   - KNN menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data baru di mana setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat dengan k adalah sebuah angka positif. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model karena jika k yang digunakan terlalu rendah maka akan menghasilkan model yang overfit dan hasil prediksi memiliki varians tinggi sedangkan jika k terlalu tinggi maka model yang dihasilkan akan underfit dan prediksinya memiliki bias tinggi.
   - Parameter yang digunakan adalah n_neighbors = 5 sebagai jumlah tetangga terdekat yang digunakan untuk prediksi dan metric Euclidean untuk mengukur jarak antara titik. Pada bagian ini, hanya akan melatih data latih dan menyimpan data uji untuk di tahap evaluasi. 

# Kelebihan:

- Mudah dipahami dan digunakan.
- Tidak memerlukan pelatihan model.
- Bekerja dengan baik pada data dengan fitur yang relatif sedikit.
  
# Kekurangan: 

- Sensitif terhadap fitur yang tidak distandarisasi.
- Tidak memberikan interpretabilitas fitur.
- Kinerja menurun jika jumlah fitur atau dimensi yang besar atau curse of dimensionality (kutukan dimensi) di mana jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) data. 

2.	Random Forest
   
   - RF adalah model machine learning yang termasuk ke dalam kategori ensemble (group) learning di mana terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah sehingga tingkat keberhasilan menjadi lebih tinggi dibandingkan yang bekerja sendiri di mana setiap model harus membuat prediksi secara independen kemudian prediksi dari setiap model digabungkan untuk membuat prediksi akhir. Algoritma RF disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fitur dilakukan secara acak.
   - Parameter yang digunakan adalah n_estimators = 100 yang merupakan jumlah trees (pohon) di forest. Lalu, ada max_depth = 5 yang merupakan kedalaman atau panjang pohon di mana merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap mode ke dalam jumlah pengamatan yang diinginkan. Selanjutnya, terdapat random_state = 42 di mana digunakan untuk mengontrol random number generator yang digunakan. Terakhir, terdapat n_jobs = -1 yang merupakan jumlah job (pekerjaan) yang digunakan secara pararel di mana merupakan komponen yang digunakan untuk mengontrol thread atau proses yang berjalan secara pararel.
     
# Kelebihan: 

-	Tangguh terhadap overfitting karena terdiri dari ensemble dari banyak pohon. 
-	Dapat menangani variabel numerik maupun kategorikal walaupun diproyek ini seluruh fitur merupakan variabel numerik. 
-	Memberikan informasi yang penting untuk melakukan intepretasi pada variabel.
  
# Kekurangan: 

-	Intepretasi pada RF lebih kompleks dibandingkan decision tree secara sendiri. 
-	Memerlukan waktu untuk melakukan komputasi yang lebih besar daripada model sedehana seperti KNN. 

3.	Boosting Algorithm
   
   - Boosting adalah algoritma yang melatih model secara berurutan atau dalam proses yang iterative di mana algoritma  yang menggunakan teknik boosting bekerja dengan membangun model dari data latih kemudian membuat model kedua yang bertugas untuk memperbaiki kesalahan dari model pertama sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Algoritma boosting digunakan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk model yang kuat (strong ensemble learner). Pada tahapan ini, metode algoritma boosting yang akan digunakan adalah adaptive boosting, salah satunya adalah AdaBoost.
   - Parameter yang digunakan adalah learning_rate = 0.05 yang merupakan bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting. Lalu, terdapat random_state = 42 yang merupakan pengontrolan random number generator yang digunakan.
     
# Kelebihan: 

-	Sangat powerful dalam meningkatkan akurasi prediksi, sering digunakan oleh para pemenang kompetisi. 
-	Dapat menangkap pola kompleks yang non-linear. 
-	Dapat ditune secara mendalam untuk mendapatkan performa yang optimal.
  
#	Kekurangan: 

-	Lebih memakan waktu saat pelatihan dibandingkan dengan algoritma RF. 
-	Sangat sensitif terhadap parameter dan dapat overfitting jika tidak dikontrol. 
-	Intepretasi yang dilakukan lebih sulit dibandingkan dengan model pohon tunggal.
  
Setelah ketiga model pada tahap pemodelan dievaluasi menggunakan data uji dengan metrik MSE, hasil yang didapat adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/a6154d7d-5a25-423c-a42e-cd60236cf182)

Berdasarkan hasil evaluasi di atas, didapatkan bahwa KNN dipilih sebagai model terbaik karena model KNN menampilkan nilai error terkecil pada data uji dengan sebesar 478.08 dibandingkan kedua algoritma lainnya. 

## Evaluation
Metrik evaluasi yang dilakukan pada tahapan ini adalah MSE (Mean Squared Error) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi karena proyek ini ingin memprediksi harga perumahan yang mana harga merupakan variabel kontinu sehingga menggunakan metrik untuk kasus regresi. MSE didefinisikan dengan persamaan berikut:

![image](https://github.com/user-attachments/assets/a2d74bcc-e550-4212-ab51-870918cdf473)

Dengan keterangan: 

-	N = jumlah dataset 
-	yi = nilai sebenarnya atau aktual
-	y_pred = nilai prediksi
  
Sebelum menghitung nilai MSE dalam model yang telah dibuat, hal yang harus dilakukan adalah melakukan scaling fitur numerik pada data uji untuk menghindari kebocoran data dengan tiga model algoritma yang telah selesai dilatih supaya skala antara data latih dan data uji menjadi sama sehingga dapat dilakukan evaluasi. Setelah melakukan scaling fitur numerik maka akan melakukan evaluasi untuk ketiga model algoritma yang telah dilatih dengan menghitung nilai MSE pada data train dan test di mana akan dibagi dengan 1e6 supaya nilai mse berada dalam skala yang tidak terlalu besar sebagai berikut:

![image](https://github.com/user-attachments/assets/426c527f-43f0-4e24-9ef6-54a5447f42dd)

Hasil evaluasi pada data latih dan data test adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/87b47291-d7ca-4110-b3b2-1e6fc405d2e6)

Untuk hasil plot metrik menggunakan bar chart adalah sebagai berikut: 

![image](https://github.com/user-attachments/assets/5bb633a3-665d-4b70-ab35-f169b3e29e48)

Berdasarkan bar plot di atas, dapat terlihat bahwa model KNN memberikan nilai error yang terkecil dibandingkan kedua model lainnya dengan model algoritma Boosting memberikan nilai error terbesar dengan di atas 1800 sedangkan untuk model RF memberikan nilai error yang tidak begitu besar dan tidak begitu kecil sehingga model KNN yang akan dipilih untuk memprediksi harga perumahan Utrecht. 

Selanjutnya, akan dilakukan pengujian dalam membuat prediksi dengan menggunakan beberapa harga dari data test yang hasilnya adalah sebagai berikut: 

![image](https://github.com/user-attachments/assets/cb0b4cf5-2ec3-4732-a725-70e0d638a802)

Berdasarkan pengujian di atas, didapatkan hasil bahwa model KNN memberikan hasil yang paling mendekati dengan y_true. 
