import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample YouTube title
title = "10 Amazing Facts About Space You Didn't Know!"

# Function to extract keywords
def extract_keywords(title):
    tokens = nltk.word_tokenize(title)
    keywords = [word for word in tokens if word.isalpha()]
    return keywords

# Function to perform sentiment analysis
def sentiment_analysis(title):
    analysis = TextBlob(title)
    return analysis.sentiment.polarity

# Function to assess readability
def readability_score(title):
    return TextBlob(title).sentiment.subjectivity

# Function to compare with top titles
def compare_with_top_titles(title, top_titles):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([title] + top_titles)
    similarity = (tfidf_matrix * tfidf_matrix.T).A[0][1:]
    return similarity.mean()

# Example top performing titles for comsparison
top_titles = [
    "Top 10 Space Facts That Will Blow Your Mind",
    "Amazing Space Discoveries You Need to Know",
    "Incredible Facts About the Universe"
]

# Extract keywords
keywords = extract_keywords(title)
print("Keywords:", keywords)

# Perform sentiment analysis
sentiment = sentiment_analysis(title)
print("Sentiment Score:", sentiment)

# Assess readability
readability = readability_score(title)
print("Readability Score:", readability)

# Compare with top titles
similarity_score = compare_with_top_titles(title, top_titles)
print("Similarity Score with Top Titles:", similarity_score)
