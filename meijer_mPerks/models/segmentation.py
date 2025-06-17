from sklearn.cluster import KMeans

def compute_rfm(transactions_df):
    # Compute recency, frequency, monetary
    rfm = ...
    return rfm

def segment_customers(rfm_df, n_clusters=5):
    km = KMeans(n_clusters)
    return km.fit_predict(rfm_df[['recency','frequency','monetary']])
