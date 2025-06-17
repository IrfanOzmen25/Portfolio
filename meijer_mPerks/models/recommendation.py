import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def build_collab_filtering(ratings_df):
    user_item = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    sim = cosine_similarity(user_item)
    return sim, user_item

def recommend(user_id, user_item, sim, top_n=10):
    idx = user_item.index.get_loc(user_id)
    scores = sim[idx] @ user_item.values / sim[idx].sum()
    top_idx = scores.argsort()[::-1][:top_n]
    return user_item.columns[top_idx]

def item_embeddings(ratings_df, n_components=50):
    svd = TruncatedSVD(n_components)
    X = svd.fit_transform(ratings_df.pivot(index='item_id', columns='user_id', values='rating').fillna(0))
    return dict(zip(ratings_df['item_id'].unique(), X))
