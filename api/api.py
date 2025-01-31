from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
import pandas as pd
import numpy as np
import os
import ast
import torch
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# ✅ Define Paths
PROJECT_PATH = r"C:\Users\SOHAM\Movie Recommendation System"
DATA_PATH = os.path.join(PROJECT_PATH, "data")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")

MOVIES_EMBEDDINGS_PATH = os.path.join(MODELS_PATH, "movies_with_embeddings.csv")
SVD_MODEL_PATH = os.path.join(MODELS_PATH, "svd_model.pkl")
NN_MODEL_PATH = os.path.join(MODELS_PATH, "movie_recommendation_model.keras")
RATINGS_PATH = os.path.join(DATA_PATH, "ratings.csv")

# ✅ Initialize FastAPI
app = FastAPI(title="Movie Recommendation API")

# ✅ Ensure Paths Exist
def check_file_exists(file_path, file_description):
    if not os.path.exists(file_path):
        raise ValueError(f"❌ {file_description} not found at {file_path}. Please ensure it is placed correctly.")

check_file_exists(SVD_MODEL_PATH, "SVD Model")
check_file_exists(NN_MODEL_PATH, "Neural Network Model")
check_file_exists(MOVIES_EMBEDDINGS_PATH, "Movies with Embeddings dataset")
check_file_exists(RATINGS_PATH, "Ratings dataset")

# ✅ Load the SVD Model
try:
    with open(SVD_MODEL_PATH, 'rb') as f:
        svd_model = pickle.load(f)
except Exception as e:
    raise ValueError(f"❌ Error loading SVD model: {e}")

# ✅ Load Neural Network Model
try:
    nn_model = tf.keras.models.load_model(NN_MODEL_PATH)
except Exception as e:
    raise ValueError(f"❌ Error loading Neural Network model: {e}")

# ✅ Load Movies Data with Space-Separated Embeddings
try:
    movies = pd.read_csv(MOVIES_EMBEDDINGS_PATH, delimiter=',', dtype=str)

    # ✅ Convert space-separated embeddings correctly
    def parse_embedding(embedding):
        try:
            # Split space-separated values and convert to NumPy array
            return np.array([float(num) for num in embedding.strip(" []").split()])
        except Exception as e:
            print(f"⚠️ Error parsing embedding: {embedding} -> {e}")
            return np.zeros(384)  # Assuming 384-dimensional embeddings
    
    # ✅ Apply fix to read embeddings correctly
    movies['embeddings'] = movies['embeddings'].apply(parse_embedding)

    # ✅ Precompute matrix for efficiency
    movie_embeddings_matrix = np.vstack(movies['embeddings'].values)

except Exception as e:
    raise ValueError(f"❌ Error loading movie dataset: {e}")

# ✅ Load Ratings Data
try:
    ratings = pd.read_csv(RATINGS_PATH, usecols=['userId', 'movieId', 'rating'])
    ratings.drop_duplicates(inplace=True)
    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
except Exception as e:
    raise ValueError(f"❌ Error loading ratings dataset: {e}")

# ✅ Load Hugging Face Model for Embeddings
MODEL_PATH = os.path.join(MODELS_PATH, "sentence-transformers_all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

# ✅ Cache to Store Query Embeddings
query_cache = {}

# ✅ Function to generate query embeddings (Optimized)
def generate_embeddings_batch(texts, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

# ✅ Optimized Content-Based Recommendation
def recommend_movies(query: str, top_n: int = 5):
    global query_cache  # ✅ Ensure query_cache is accessible

    # ✅ Check if query already exists in cache
    if query in query_cache:
        query_embedding = query_cache[query]
    else:
        query_embedding = generate_embeddings_batch([query])[0].reshape(1, -1)  # ✅ Generate new embedding
        query_cache[query] = query_embedding  # ✅ Store in cache for future use

    # ✅ Compute Similarities
    similarities = cosine_similarity(query_embedding, movie_embeddings_matrix)[0]
    
    # ✅ Store similarities and get recommendations
    movies["similarity"] = similarities
    recommendations = movies.sort_values(by="similarity", ascending=False).head(top_n)
    
    return recommendations[['title', 'similarity']].to_dict(orient="records")


# ✅ SVD-Based Recommendation
def recommend_movies_svd(user_id: int, top_n: int = 5):
    if user_id not in ratings['userId'].unique():
        raise HTTPException(status_code=400, detail="❌ User ID not found in ratings dataset.")
    
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movie_ids = movies['id'].unique()
    unrated_movies = [movie for movie in all_movie_ids if movie not in user_rated_movies]

    predictions = [(movie, svd_model.predict(user_id, movie).est) for movie in unrated_movies]
    top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommended_movies = pd.DataFrame({
        "movieId": [movie for movie, _ in top_predictions],
        "predicted_rating": [rating for _, rating in top_predictions]
    })

    recommended_movies = recommended_movies.merge(
        movies[['id', 'title', 'genres_str', 'release_year']].rename(columns={'id': 'movieId'}),
        on='movieId'
    )

    return recommended_movies[['title', 'predicted_rating', 'release_year']].to_dict(orient="records")

# ✅ Hybrid Recommendation
def hybrid_recommendation(user_id: int, genre: str, top_n: int = 5):
    if user_id not in ratings['userId'].unique():
        raise HTTPException(status_code=400, detail="❌ User ID not found in ratings dataset.")

    # ✅ Get SVD-based recommendations
    svd_recommendations = recommend_movies_svd(user_id, top_n * 2)

    # ✅ Get Content-Based Recommendations using genre
    genre_movies = movies[movies['genres_str'].str.contains(genre, case=False, na=False)]
    genre_movies = genre_movies.drop_duplicates(subset='id')
    top_genre_movies = genre_movies.sort_values(by='popularity', ascending=False).head(top_n * 2)

    # ✅ Merge SVD and Content-Based Recommendations
    combined = pd.merge(pd.DataFrame(svd_recommendations), top_genre_movies[['title', 'popularity']], on='title', how='outer')
    combined['predicted_rating'] = combined['predicted_rating'].fillna(0)
    combined['popularity'] = combined['popularity'].fillna(0)

    combined['hybrid_score'] = combined['predicted_rating'] * 0.7 + combined['popularity'] * 0.3
    return combined.sort_values(by='hybrid_score', ascending=False).head(top_n).to_dict(orient="records")

# ✅ API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommendation System!"}

@app.get("/recommendations/content/")
def content_recommendation(query: str, top_n: int = 5):
    return recommend_movies(query, top_n)

@app.get("/recommendations/svd/")
def svd_recommendation(user_id: int, top_n: int = 5):
    return recommend_movies_svd(user_id, top_n)

@app.get("/recommendations/hybrid/")
def hybrid_movie_recommendation(user_id: int, genre: str, top_n: int = 5):
    return hybrid_recommendation(user_id, genre, top_n)

# ✅ Run API
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
