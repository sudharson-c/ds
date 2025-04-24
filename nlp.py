import nltk
import string
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens

def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

doc1 = read_file('doc1.txt')
doc2 = read_file('doc2.txt')

tokens_doc1 = tokenize_text(doc1)
tokens_doc2 = tokenize_text(doc2)

sentiment_doc1 = sentiment_analysis(doc1)
sentiment_doc2 = sentiment_analysis(doc2)

similarity_score = calculate_similarity(doc1, doc2)

generate_word_cloud(doc1)
generate_word_cloud(doc2)

print("Tokens in Document 1:", tokens_doc1)
print("Tokens in Document 2:", tokens_doc2)
print(f"\nSentiment of Document 1: {sentiment_doc1}")
print(f"Sentiment of Document 2: {sentiment_doc2}")
print(f"\nSimilarity Score between Document 1 and Document 2: {similarity_score}")

