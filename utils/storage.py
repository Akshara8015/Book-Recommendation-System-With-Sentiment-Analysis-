import pandas as pd
import numpy as np
import re
import pickle
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

import warnings
warnings.filterwarnings("ignore")
import requests
import urllib.parse
from bs4 import BeautifulSoup


# Load once (global)
tf = joblib.load("model/tfidf.pkl")
vectors = joblib.load("model/vectors.pkl")
df = joblib.load("model/books_df.pkl")

# books_review_page_links = {}
# books_reviews = {}
# books_profile_reviews = {}
# book_rating = {}
# book_img = {}

# def load_data():
#     global books_review_page_links, books_reviews
#     global books_profile_reviews, book_rating, book_img

#     try:
#         with open("data/data.pkl", "rb") as f:
#             data = pickle.load(f)

#             books_review_page_links = data.get("links", {})
#             books_reviews = data.get("reviews", {})
#             books_profile_reviews = data.get("profile", {})
#             book_rating = data.get("rating", {})
#             book_img = data.get("images", {})

#     except:
#         pass


# def save_data():
#     data = {
#         "links": books_review_page_links,
#         "reviews": books_reviews,
#         "profile": books_profile_reviews,
#         "rating": book_rating,
#         "images": book_img
#     }

#     with open("data/data.pkl", "wb") as f:
#         pickle.dump(data, f)
        


def recommend(movie_name):
    book_idx = df[df['title'] == movie_name].index[0]
    distance = cosine_similarity(vectors[book_idx], vectors).flatten()
    books_list = sorted(list(enumerate(distance)), key=lambda x: x[1], reverse=True)[1:6]

    books = []

    for idx, _ in books_list:
        book_name = df.iloc[idx]['title']
        author = df.iloc[idx]['author']
        books.append(book_name + ' by ' +  author)
    return books


def get_book_page(book_name_with_author):
    query = urllib.parse.quote(book_name_with_author)
    URL = f"https://www.amazon.com/s?k={query}&i=stripbooks-intl-ship&crid=PLVOGG581S7X&sprefix=the+waves+%2Cstripbooks-intl-ship%2C514&ref=nb_sb_noss_2"
    header = {
        "User-Agent": "Mozilla/5.0" ,
        "referer" : URL,
        "Accept-Language" : "en-US,en;q=0.5"
    }
    response = requests.get(URL, headers = header)
    print(response.status_code)
    soup = BeautifulSoup(response.text, "lxml")
    link = soup.find("a", attrs = {
        "class" : "a-link-normal s-line-clamp-2 puis-line-clamp-3-for-col-4-and-8 s-link-style a-text-normal"
    })
    link = "https://www.amazon.com/" + link.get("href")
    return link


def get_review_page(link):
    header = {
        "User-Agent": "Mozilla/5.0" ,
        "referer" : link,
        "Accept-Language" : "en-US,en;q=0.5"
    }
    response = requests.get(link, headers = header)
    soup = BeautifulSoup(response.text, "lxml")
    profiles = soup.find_all("div", attrs = {"class" : "a-row a-spacing-mini"})
    reviews = soup.find_all("div", attrs = {"class" : "a-expander-content reviewText review-text-content a-expander-partial-collapse-content"})
    img_link = soup.find("img", attrs = {"class" : "a-dynamic-image a-stretch-vertical media-block-image-tag"})
    print(reviews)

    # profile_with_review = {}
    # for i in range(len(reviews)):
    #     profile_with_review[profiles[i]] = reviews[i].text.split()
    # books_profile_reviews[book_name_with_author] = profile_with_review

    r = []
    for x in reviews:
        r.append(x.text.strip())
    print(len(reviews))
    return img_link , r
    # return img_link.get('src') , r


# Use SAME vectorizer
def reviews_to_stars(reviews):
    if reviews:
        X_new = tf.transform(reviews)
        X_new = X_new.toarray()
        X_tensor = torch.tensor(X_new, dtype=torch.float32)

        rnn_model = RNN(input_size=5000)
        rnn_model.load_state_dict(torch.load("model/rnn_model.pth"))
        rnn_model.eval()

        with torch.no_grad():
            output = rnn_model(X_tensor.unsqueeze(1))  # (batch,1,5000)
            probs = torch.sigmoid(output)
        avg_score = probs.mean().item()
        return np.round(avg_score * 5, 2)
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size = 128, no_of_layers = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.no_of_layers = no_of_layers

        self.rnn = nn.RNN(input_size, hidden_size, no_of_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.no_of_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:,-1,:])

        return out
    