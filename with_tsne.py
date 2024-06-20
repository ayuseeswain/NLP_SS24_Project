from data_processing import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


y_album = merged_df['Album']
y_year = merged_df['Year']

# Topic Modeling - NMF
nmf_model = NMF(n_components=10)
nmf_topics = nmf_model.fit_transform(tfidf_matrix)

# Topic Modeling - LDA
lda_model = LatentDirichletAllocation(n_components=10)
lda_topics = lda_model.fit_transform(tfidf_matrix)


# Combine NMF and LDA features
X_combined = np.hstack((nmf_topics, lda_topics))

# Encode the target variables
label_encoder_album = LabelEncoder()
label_encoder_year = LabelEncoder()
y_album_encoded = label_encoder_album.fit_transform(y_album)
y_year_encoded = label_encoder_year.fit_transform(y_year)

# Balance the classes using SMOTE
smote = SMOTE(random_state=42)
X_combined_balanced, y_album_balanced = smote.fit_resample(X_combined, y_album_encoded)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_combined_balanced)

# Plot t-SNE results
plt.figure(figsize=(12, 8))
for album in np.unique(y_album_balanced):
    indices = np.where(y_album_balanced == album)
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label_encoder_album.inverse_transform([album])[0], alpha=0.5)
plt.legend()
plt.title('t-SNE visualization of combined NMF and LDA features')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()

# Applying PCA to the balanced datasets
pca = PCA(n_components=20, random_state=42)
X_combined_pca = pca.fit_transform(X_combined_balanced)

# Train-test split with stratification
X_train_combined, X_test_combined, y_train_album, y_test_album = train_test_split(X_combined_pca, y_album_balanced, test_size=0.2, random_state=42, stratify=y_album_balanced)

# Random Forest for Album Prediction
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

rf_clf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train_combined, y_train_album)
best_rf_model = rf_grid_search.best_estimator_

# Predicting with the best Random Forest model
y_pred_rf = best_rf_model.predict(X_test_combined)

# Decode the predictions back to original labels
y_pred_rf_decoded = label_encoder_album.inverse_transform(y_pred_rf)

# Evaluate the Random Forest model performance
def evaluate_model(y_true, y_pred, task_name):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f"Performance metrics for {task_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}\n")

evaluate_model(y_test_album, y_pred_rf, 'Album Prediction (Combined NMF+LDA) with Random Forest')

# Combine NMF and LDA features for year prediction
X_combined_year = np.hstack((nmf_topics, lda_topics))

# Balance the classes using SMOTE for year prediction
X_combined_year_balanced, y_year_balanced = smote.fit_resample(X_combined_year, y_year_encoded)

# Applying PCA to the balanced datasets for year prediction
X_combined_year_pca = pca.fit_transform(X_combined_year_balanced)

# Train-test split with stratification for year prediction
X_train_combined_year, X_test_combined_year, y_train_year, y_test_year = train_test_split(X_combined_year_pca, y_year_balanced, test_size=0.2, random_state=42, stratify=y_year_balanced)

# Random Forest for Year Prediction
rf_grid_search_year = GridSearchCV(rf_clf, rf_param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
rf_grid_search_year.fit(X_train_combined_year, y_train_year)
best_rf_model_year = rf_grid_search_year.best_estimator_

# Predicting with the best Random Forest model for year
y_pred_rf_year = best_rf_model_year.predict(X_test_combined_year)

# Decode the predictions back to original labels for year
y_pred_rf_year_decoded = label_encoder_year.inverse_transform(y_pred_rf_year)

# Evaluate the Random Forest model performance for year
evaluate_model(y_test_year, y_pred_rf_year, 'Year Prediction (Combined NMF+LDA) with Random Forest')

# Example: Adding a thematic category column
thematic_categories = {
    'Fearless': 'Love',
    '1989': 'Heartbreak',
    'Red': 'Relationships',
    'Lover': 'Happiness',
    'Folklore': 'Melancholy',
    'Evermore': 'Nostalgia',
    'Reputation': 'Revenge',
    'Midnights': 'Introspection',
    'Taylor Swift': 'Coming of Age',
    'Speak Now': 'Growth'
}

# Map thematic categories to each album
merged_df['Thematic_Category'] = merged_df['Album'].map(thematic_categories)

# Encode the target variable for thematic category
label_encoder_thematic_category = LabelEncoder()
y_thematic_category = merged_df['Thematic_Category']
y_thematic_category_encoded = label_encoder_thematic_category.fit_transform(y_thematic_category)

# Balance the classes using SMOTE
X_combined_thematic_balanced, y_thematic_balanced = smote.fit_resample(X_combined, y_thematic_category_encoded)

# Applying PCA to the balanced datasets
X_combined_thematic_pca = pca.fit_transform(X_combined_thematic_balanced)

# Train-test split with stratification
X_train_combined_thematic, X_test_combined_thematic, y_train_thematic, y_test_thematic = train_test_split(X_combined_thematic_pca, y_thematic_balanced, test_size=0.2, random_state=42, stratify=y_thematic_balanced)

# Random Forest for Thematic Category Prediction
rf_grid_search_thematic = GridSearchCV(rf_clf, rf_param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
rf_grid_search_thematic.fit(X_train_combined_thematic, y_train_thematic)
best_rf_model_thematic = rf_grid_search_thematic.best_estimator_

# Predicting with the best Random Forest model
y_pred_rf_thematic = best_rf_model_thematic.predict(X_test_combined_thematic)

# Decode the predictions back to original labels
y_pred_rf_thematic_decoded = label_encoder_thematic_category.inverse_transform(y_pred_rf_thematic)

# Evaluate the Random Forest model performance
evaluate_model(y_test_thematic, y_pred_rf_thematic, 'Thematic Category Prediction (Combined NMF+LDA) with Random Forest')

def predict_album_from_lyrics(lyrics):
    # Preprocess lyrics
    lyrics_cleaned = clean_lyrics(lyrics)
    lyrics_no_stopwords = remove_stopwords(lyrics_cleaned)
    lyrics_processed = tokenize_and_lemmatize(lyrics_no_stopwords)
    
    # Extract features
    nmf_topics_lyrics = nmf_model.transform(tfidf_vectorizer.transform([lyrics_processed]))
    lda_topics_lyrics = lda_model.transform(tfidf_vectorizer.transform([lyrics_processed]))
    combined_topics_lyrics = np.hstack((nmf_topics_lyrics, lda_topics_lyrics))
    
    # Apply PCA
    combined_topics_pca_lyrics = pca.transform(combined_topics_lyrics)
    
    # Predict the album using the trained model
    predicted_album_encoded = best_rf_model.predict(combined_topics_pca_lyrics)
    predicted_album = label_encoder_album.inverse_transform(predicted_album_encoded)
    
    print(f"The predicted album for the given lyrics is: {predicted_album[0]}")

def predict_year_from_album(album_name):
    # Find the album in the dataset
    album_data = merged_df[merged_df['Album'] == album_name]
    
    if album_data.empty:
        print("Album not found in the dataset.")
        return
    
    # Combine NMF and LDA features
    album_nmf_topics = nmf_model.transform(tfidf_vectorizer.transform(album_data['processed_lyrics']))
    album_lda_topics = lda_model.transform(tfidf_vectorizer.transform(album_data['processed_lyrics']))
    album_combined_topics = np.hstack((album_nmf_topics, album_lda_topics))
    
    # Apply PCA
    album_combined_pca = pca.transform(album_combined_topics)
    
    # Predict the year using the trained model
    predicted_year_encoded = best_rf_model_year.predict(album_combined_pca)
    predicted_year = label_encoder_year.inverse_transform(predicted_year_encoded)
    
    print(f"The predicted release year for the album '{album_name}' is: {predicted_year[0]}")

def predict_thematic_category_from_lyrics(lyrics):
    # Preprocess lyrics
    lyrics_cleaned = clean_lyrics(lyrics)
    lyrics_no_stopwords = remove_stopwords(lyrics_cleaned)
    lyrics_processed = tokenize_and_lemmatize(lyrics_no_stopwords)
    
    # Extract features
    nmf_topics_lyrics = nmf_model.transform(tfidf_vectorizer.transform([lyrics_processed]))
    lda_topics_lyrics = lda_model.transform(tfidf_vectorizer.transform([lyrics_processed]))
    combined_topics_lyrics = np.hstack((nmf_topics_lyrics, lda_topics_lyrics))
    
    # Apply PCA
    combined_topics_pca_lyrics = pca.transform(combined_topics_lyrics)
    
    # Predict the thematic category using the trained model
    predicted_thematic_encoded = best_rf_model_thematic.predict(combined_topics_pca_lyrics)
    predicted_thematic = label_encoder_thematic_category.inverse_transform(predicted_thematic_encoded)
    
    print(f"The predicted thematic category for the given lyrics is: {predicted_thematic[0]}")

# Test the functions
predict_album_from_lyrics("We were both young when I first saw you...")
predict_year_from_album("Fearless")
predict_thematic_category_from_lyrics("We were both young when I first saw you...")
