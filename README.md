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
- Release Year  
- Thematic Category (e.g., Nostalgia, Growth)

---

## 🔍 Methods Overview

### 1. Data Collection & Preprocessing
- Collected album metadata (`Album_Lists.csv`) and loaded song lyrics from text files.
- Applied formatting to align track names with lyric file naming conventions.
- Lyrics were cleaned via:
  - Removal of metadata, filler words, special characters
  - Tokenization using `nltk.word_tokenize`
  - Stopword removal and lemmatization (`WordNetLemmatizer`)

### 2. Feature Extraction
- Applied **TF-IDF Vectorization** (`TfidfVectorizer`) on cleaned lyrics with n-grams up to trigrams.
- Reduced dimensionality using **PCA**.
- Balanced data with **SMOTE**.

### 3. Topic Modeling
- Extracted 20 latent topics with both NMF and LDA.
- Visualized:
  - Top words per topic
  - Aggregate topic weights
  - Album-wise topic distribution
  - Word clouds per album and overall

### 4. Sentiment Analysis
- Used **TextBlob** to assign a polarity score (from -1 to +1) to each track's lyrics.
- Visualized sentiment distribution with **Plotly** (by album, year, and theme).

### 5. Predictive Modeling
- Models trained to classify:
  - Album (multi-class)
  - Year (multi-class)
  - Thematic category (multi-class)
- Hyperparameters optimized using `RandomizedSearchCV` over 5-fold `StratifiedKFold` splits.
- Features: NMF topic vectors → PCA → standardized (for SVM)

---

## 📊 Model Comparison

### NMF vs. LDA
- NMF produced **more balanced** and **distinct** topic distributions.
- LDA was more prone to **single-topic domination**.

### Random Forest vs. SVM
- Random Forest yielded **better accuracy and generalization**, especially for overlapping themes.
- SVM tended to **overfit** and sometimes mapped different inputs to the same output.

---

## 🎨 Visualizations
- Word Clouds for each album
- Topic distributions stacked by album
- Sentiment vs Year scatter plots
- Classifier performance bar charts (Accuracy, F1, Precision, Recall)

---

## 🧪 Results & Insights
Random Forest, when paired with NMF topic vectors, provided the most accurate predictions for album, year, and theme.

---

## 🧪 Example Input and Output

**Input Lyrics:**
> We were both young when I first saw you / I close my eyes and the flashback starts...

**Predicted Output:**
- **Album:** Fearless  
- **Year:** 2008  
- **Theme:** Romantic Nostalgia

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

## 🧠 Conclusion

This project demonstrates how combining topic modeling (NMF) with ensemble learning (Random Forest) can yield robust pipelines for text analysis and classification. While SVM suffered from overfitting, Random Forest handled complex relationships better. The findings highlight how lyrical themes can be linked to albums, time periods, and artistic growth.

