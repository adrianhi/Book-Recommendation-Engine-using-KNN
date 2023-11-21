# Import necessary libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the dataset
books = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")
ratings = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")

# Merge the books and ratings dataframes
df = pd.merge(ratings, books, on='book_id')

# Filter out users with less than 200 ratings and books with less than 100 ratings
user_ratings_counts = df['user_id'].value_counts()
book_ratings_counts = df['book_id'].value_counts()

df = df[df['user_id'].isin(user_ratings_counts[user_ratings_counts >= 200].index)]
df = df[df['book_id'].isin(book_ratings_counts[book_ratings_counts >= 100].index)]

# Create a pivot table
book_matrix = df.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

# Build the Nearest Neighbors model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
knn.fit(book_matrix.values.T)

# Function to get book recommendations
def get_recommends(book_title):
    book_idx = book_matrix.index.get_loc(book_title)
    distances, indices = knn.kneighbors([book_matrix.iloc[book_idx].values], 6)

    recommended_books = []
    for i in range(1, len(distances.flatten())):
        recommended_books.append([book_matrix.index[indices.flatten()[i]], distances.flatten()[i]])

    return [book_title, recommended_books]

# Test the function
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
