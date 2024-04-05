import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import linear_kernel # type: ignore

# Sample movie dataset
data = {
    'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
    'genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama'],
    'description': [
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'An organized crime dynasty\'s aging patriarch transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN values with empty string
df['description'] = df['description'].fillna('')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on user preferences
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]

# Example usage
user_preference = 'The Shawshank Redemption'  # User's preferred movie
recommended_movies = recommend_movies(user_preference)
print("Recommended Movies for You:")
print(recommended_movies)
