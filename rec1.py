# we have done recommendation system for  items
import numpy as np
from scipy.linalg import svd
from numpy.linalg import norm
# Generate temporary data
matrix = np.array([[5, 3, 0, 1,], [4, 0, 0, 1], [ 1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4]])
items = ['item1', 'item2', 'item3', 'item4']
print("Original matrix:\n" ,matrix)
U, S, V = svd(matrix, full_matrices=False)
k = 3  #key features to keep/ reduced dimensions
print("\nU:-",U)
print("\nS:-",S)
print("\nV:-",V)

#------------------------------------

matrix_svd = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
print("SVD Matrix:\n", matrix_svd)


def item_similarity(matrix):
    similarity = np.dot(matrix.T,matrix)/(norm(matrix.T)*norm(matrix))
    np.fill_diagonal(similarity, 0)
    return similarity
similarity_svd = item_similarity(matrix_svd)
print("similarity :\n",similarity_svd)


#----------------------
# recommendation part 

def recommend(user, matrix, similarity, items):
    user_index = user - 1
    user_ratings = matrix[user_index, :]
    item_scores = np.zeros((matrix.shape[1],))
    for item in range(matrix.shape[1]):
        item_sum = 0
        for other_item in range(matrix.shape[1]):
            if user_ratings[other_item] == 0 or similarity[item][other_item] == 0:
                continue
            item_sum = user_ratings[other_item] * similarity[item][other_item]
        item_scores[item] = item_sum / np.abs(similarity[item]).sum()
    recommendations = [items[i] for i in np.argsort(item_scores)[::-1]]
    print(item_scores)
    return recommendations

n=int(input("Enter the user: "))
user =n
recommendations = recommend(user, matrix, similarity_svd, items)
print(f"Recommendations for user {user}: {recommendations}")
