
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1 - Load the data
df = pd.read_csv(r'D:\task2 recommendation\ml-100k\u.data', sep='\t',
                 names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movie titles (moved to top so it's available everywhere)
movies = pd.read_csv(r'D:\task2 recommendation\ml-100k\u.item',
                     sep='|', encoding='latin-1',
                     usecols=[0, 1],
                     names=['movie_id', 'title'])

print("Data loaded successfully!")
print(df.head())

# Step 2 - Build the user-item matrix
matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')
matrix = matrix.fillna(0)

print("\nMatrix shape:", matrix.shape)
print("Matrix ready!")

# Step 3 - Calculate similarity between users
user_similarity = cosine_similarity(matrix)
user_sim_df = pd.DataFrame(user_similarity,
                            index=matrix.index,
                            columns=matrix.index)

print("User similarity calculated!")

# Step 4 - Recommend movies for a user
def recommend_movies(user_id, num_recommendations=5):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    top_similar = similar_users.head(5).index.tolist()

    watched = matrix.loc[user_id]
    not_watched = watched[watched == 0].index.tolist()

    scores = {}
    for movie in not_watched:
        score = 0
        for sim_user in top_similar:
            score += matrix.loc[sim_user, movie] * user_sim_df.loc[user_id, sim_user]
        scores[movie] = score

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return recommended[:num_recommendations]

# Ask for any user ID
user_id = int(input("\nEnter a user ID (1 to 943): "))
recommendations = recommend_movies(user_id)

print(f"\nTop 5 movie recommendations for User {user_id}:")
for i, (movie_id, score) in enumerate(recommendations, 1):
    title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print(f"  {i}. {title}  |  Score: {score:.2f}")