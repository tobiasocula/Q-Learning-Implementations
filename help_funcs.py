import numpy as np


def k_means(data, k):
    # data: np-array of vectors (points), shape (n_vectors, vector_dim)
    # k: amount of clusters

    # data size: (n_points, vector_dim)

    center_indices = np.random.randint(0, len(data), size=(1,k)) # (1, k)
    centers = np.array([data[i] for i in center_indices]).reshape((k, data.shape[1])) # (k, vector_dim)
    
    clusters = [[] for _ in range(k)] # (k, n_vectors_to_cluster, vector_dim)
    
    while True:
        new_clusters = clusters
        """
        for every point: determine dist to every cluster
        categorize points based on dist (take smallest)
        redetermine clusters
        """

        labels = [] # length n_vectors

        # determine clusters
        for vector in data:
            # vector: (1, vector_dim)

            # distances for vector to every cluster vector
            # (1, k)

            distances = np.array([np.linalg.norm(vector - core) for core in centers])
            idx = np.argmin(distances)
            new_clusters[idx].append(vector) # list of length k
            labels.append(int(idx))

        if new_clusters == clusters:
            return clusters, centers, labels

        # determine new centers
        for k_idx, cluster in enumerate(clusters):
            # cluster: list of vectors
            avg_vector = np.zeros(data.shape[1])
            for vec in cluster:
                # vec: (vector_dim)
                avg_vector += vec
            avg_vector /= len(cluster)
            centers[k_idx,:] = avg_vector

def generate_volatility(V0, xi, kappa, theta, N, dt):
    V = np.empty(N)
    V[0] = V0
    for i in range(1, N):
        dw_v = np.sqrt(dt) * np.random.normal()
        sigma = max(0, V[i-1] + kappa * (theta - V[i-1]) * dt + xi * np.sqrt(V[i-1]) * dw_v)
        V[i] = sigma
    return V

def generate_prices(vola_values, S0, mu, dt, N):
    S = np.empty(N)
    S[0] = S0
    for i in range(1, N):
        dw_s = np.sqrt(dt) * np.random.normal()
        s = S[i-1] * np.exp((mu - 0.5 * vola_values[i]) * dt + np.sqrt(vola_values[i]) * dw_s)
        S[i] = s
    return S