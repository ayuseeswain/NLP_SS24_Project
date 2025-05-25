# 🎤 Topic Modeling on Taylor Swift Songs

This project explores the themes in Taylor Swift's lyrics using **topic modeling** and **predictive modeling** techniques. We aim to identify prominent themes in her songs, understand how these vary across different albums and periods, and build models that can predict the album, release year, or thematic category based on the lyrics.

---

## 🧠 Techniques Used

### 📌 Topic Modeling
We used two unsupervised algorithms:
- **Latent Dirichlet Allocation (LDA):** A probabilistic model that learns topic distributions over documents.
- **Non-negative Matrix Factorization (NMF):** A matrix decomposition method that reduces data into interpretable lower-dimensional topics.

### 📌 Predictive Modeling
We trained two models:
- **Random Forest**
- **Support Vector Machine (SVM)**

These models were tested on their ability to predict:
- Album
- Year
- Theme

---

## 🔍 Methods Overview

### 1. Data Collection & Preprocessing
- Compiled a list of all Taylor Swift songs and their lyrics.
- Removed irrelevant information and standardized lyrics through tokenization and lemmatization.

### 2. Feature Extraction
- Used **CountVectorizer** and **TF-IDF** to convert lyrics to numerical representations.
- Visualized word clouds per album to observe frequent keywords.

### 3. Topic Modeling
- Applied LDA and NMF to extract latent themes.
- Analyzed the theme distribution across albums using 20 topics.

### 4. Predictive Modeling
- Used extracted features to train **Random Forest** and **SVM** models.
- Performed classification tasks using metrics like **accuracy**, **precision**, and **F1-score**.
- Applied **SMOTE** for class balancing and **PCA** for dimensionality reduction.

### 5. Model Evaluation
- Compared the performance of NMF vs. LDA and Random Forest vs. SVM.
- Random Forest outperformed SVM across all prediction tasks.

---

## 📊 Model Comparison

### NMF vs. LDA
- NMF produced **more balanced** and **distinct** topic distributions.
- LDA was more prone to **single-topic domination**.

### Random Forest vs. SVM
- Random Forest yielded **better accuracy and generalization**, especially for overlapping themes.
- SVM tended to **overfit** and sometimes mapped different inputs to the same output.

---

## 🧪 Results & Insights
Random Forest, when paired with NMF topic vectors, provided the most accurate predictions for album, year, and theme.

---

## 📦 Requirements

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `spacy`
- `gensim`
- `wordcloud`
- `imbalanced-learn` (for SMOTE)

---

## 🧪 Example Input and Output

**Input Lyrics:**
> We were both young when I first saw you / I close my eyes and the flashback starts...

**Predicted Output:**
- **Album:** Fearless  
- **Year:** 2008  
- **Theme:** Romantic Nostalgia

---

## 🧠 Conclusion

This project demonstrates how combining topic modeling (NMF) with ensemble learning (Random Forest) can yield robust pipelines for text analysis and classification. While SVM suffered from overfitting, Random Forest handled complex relationships better. The findings highlight how lyrical themes can be linked to albums, time periods, and artistic growth.

---
