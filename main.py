import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("movie.csv")

print(df.head(5))
df.shape
features = ['keywords', 'cast', 'genres', 'director']
df[features].head(5)

df[features].isnull().values.any()
for feature in features:
    df[feature] = df[feature].fillna('')

df[features].isnull().values.any()

def combine_features(row):
    return row['keywords'] + ' ' + row['cast'] + ' ' + row['genres']+' '+row['director']

df['combined_features'] = df.apply(combine_features, axis =1)
df.head(3)

count_matrix = CountVectorizer().fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)
cosine_sim.shape

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_user_likes = input("Enter a movie you like : ")
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))
print(similar_movies)
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)[1:]
i = 0
print("Top 7  similar movies to " + movie_user_likes + "are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i = i+1
    if i>=7:
        break

