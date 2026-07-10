# Apar — İstifadəçi Rəylərinin Təsnifatı

> DataRace yarışı: Velosiped və skuter icarə platforması üçün çoxdilli rəy təsnifatı

## 📋 Layihə Haqqında

Bu layihə **Apar** platformasına daxil olan çoxdilli istifadəçi rəylərini avtomatik olaraq üç kateqoriyaya təsnif edir:

| Etiket | Məna |
|--------|------|
| `technical_support` | Velosipedin texniki/mexaniki nasazlığı haqqında rəy |
| `customer_support` | Xidmət, ödəniş, yolda qalma, hesab məsələləri |
| `other` | Yuxarıdakı iki kateqoriyaya uyğun gəlməyən rəy |

Rəylər **Azərbaycan**, **Rus** və **İngilis** dillərinin qarışığında yazılmışdır. Mətnlər qeyri-rəsmi üslubda ola bilər, orfoqrafik səhvlər, emojilər və ya gürültülü məlumatlar ehtiva edə bilər.

---

## 📁 Repo Strukturu

```
Apar/
├── data/                  # CSV faylları (yerli saxlanılır — GitHub-a yüklənmir)
│   ├── train.csv
│   ├── test.csv
│   └── sample.csv
├── notebooks/
│   ├── 01_eda.ipynb       # Kəşfedici məlumat analizi
│   ├── 02_preprocessing.ipynb  # Mətn ön emalı
│   └── 03_model.ipynb     # Model öyrətmə və qiymətləndirmə
├── src/
│   ├── preprocess.py      # Mətn temizleme funksiyaları
│   ├── features.py        # Feature extraction
│   └── predict.py         # Inference / submission faylı yaratma
├── outputs/
│   └── submission.csv     # Yarış üçün son submission (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗂️ Məlumat Faylları

> ⚠️ **Qeyd**: Yarış qaydalarına görə datasetlər bu repoya yüklənmir. Faylları DataRace platformasından endirin və `data/` qovluğuna əlavə edin.

| Fayl | Təsvir |
|------|--------|
| `train.csv` | Etiketlənmiş təlim dəsti (`id`, `feedback`, `tag`, `label`) |
| `test.csv` | Etiketlənməmiş test dəsti (`id`, `feedback`, `tag`) |
| `sample.csv` | Submission format nümunəsi (`id`, `label`) |

---

## 🚀 Başlanğıc

### 1. Asılılıqları qur

```bash
pip install -r requirements.txt
```

### 2. Məlumatları yerləşdir

```bash
# DataRace-dən endirdikdən sonra:
cp ~/Downloads/train.csv data/
cp ~/Downloads/test.csv data/
cp ~/Downloads/sample.csv data/
```

### 3. Notebook-ları ardıcıl icra et

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📊 Qiymətləndirmə

Yarış **macro F1-score** metrikası ilə qiymətləndirilir. Nəticə ictimai (public) və gizli (private) liderlər cədvəli üzrə bölünür.

---

## ⚙️ Texniki Məhdudiyyətlər (Qaydalar)

- ✅ Neural + ənənəvi ML metodlarına icazə var
- ✅ Xarici pretrained modellər istifadə oluna bilər (maks. **600M parametr**)
- ✅ Ansambl / hibrid yanaşmalar icazəlidir
- ⚠️ Inference yalnız **CPU** mühitində işləməlidir (maks. **8 GB RAM**)
- ⚠️ Test dataseti öyrətmə üçün istifadə **qadağandır**
- ⚠️ Ödənişli API/bulud xidmətləri inference-də **qadağandır**
- ⚠️ Datasetləri ictimai platformalara yükləmək **qəti qadağandır**
- ✅ Reproduksiya üçün `random seed` sabit olmalıdır

---

## 📦 Kitabxanalar

Baxın: [`requirements.txt`](requirements.txt)

---

## 👤 Müəllif

**Imran Hajiyev** — [GitHub](https://github.com/1mranhajiyev)
