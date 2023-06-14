import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import spacy

# Load the pre-trained English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Load the collected data into a pandas DataFrame
data = pd.read_csv('neurotechnology_papers.csv')

# Data Cleaning
# Remove duplicates
data.drop_duplicates(inplace=True)

# Handling missing values (if any)
data.dropna(inplace=True)

# Text Cleaning
data['cleaned_abstract'] = data['Abstract'].apply(lambda x: " ".join([w.lower() for w in word_tokenize(str(x))]))

# Stopword Removal
stop_words = set(stopwords.words('english'))
data['cleaned_abstract'] = data['cleaned_abstract'].apply(lambda x: " ".join([w for w in word_tokenize(x) if w.lower() not in stop_words]))

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['cleaned_abstract'] = data['cleaned_abstract'].apply(lambda x: " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(x)]))

# Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf_vectorizer.fit_transform(data['cleaned_abstract'])

# Feature Selection using chi-square test
select_k_best = SelectKBest(chi2, k=100)
selected_features = select_k_best.fit_transform(tfidf_features, data['Citations'])

# Get the selected feature names
feature_names = tfidf_vectorizer.get_feature_names_out()[select_k_best.get_support()]

# Create a new DataFrame with the selected features
selected_data = pd.DataFrame(selected_features.toarray(), columns=feature_names)

# Add the target variable (Citations) to the selected data
selected_data['Citations'] = data['Citations']
selected_data['Title'] = data['Title']

# Save the preprocessed and selected data to a new CSV file
selected_data.to_csv('preprocessed_neurotechnology_papers.csv', index=False)
