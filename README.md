Recommendation System using SVD

Overview:
Implements collaborative filtering using Singular Value Decomposition (SVD).
Generates recommendations based on user-item interactions.
Code Highlights:

Data Generation:
Temporary matrix represents user-item interactions.

SVD Decomposition:
Decomposes the matrix into U, S, and V using SciPy.

Matrix Reconstruction:
Reconstructs the matrix using a reduced set of key features (k).

Item Similarity:
Computes item similarity using dot product of transposed reconstructed matrix.

Recommendation Function:
Generates item scores based on user ratings and similarity.
Recommends items by sorting scores.

Usage:
Run the script and input a user ID.
System provides recommended items for the user.
