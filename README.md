## Table of Contents

### [1. Why are we here?](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb) 
We start with a high level overview of some foundational concepts in numerical linear algebra.
  - [Matrix and Tensor Products](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Matrix-and-Tensor-Products)
  - [Matrix Decompositions](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Matrix-Decompositions)
  - [Accuracy](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Accuracy)
  - [Memory use](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Memory-Use)
  - [Speed](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Speed)
  - [Parallelization & Vectorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Vectorization)

### [2. Background Removal with SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb)
Another application of SVD is to identify the people and remove the background of a surveillance video.
 - [Load and View Video Data](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Load-and-Format-the-Data)
  - [SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Singular-Value-Decomposition)
  - [Making a video](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Make-Video)
  - [Speed of SVD for different size matrices](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Speed-of-SVD-for-different-size-matrices)
  - [Two backgrounds in one video](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#2-Backgrounds-in-1-Video)
  - [Data compression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Aside-about-data-compression)
  
### [3. Topic Modeling with NMF and SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb) 
We will use the newsgroups dataset to try to identify the topics of different posts.  We use a term-document matrix that represents the frequency of the vocabulary in the documents.  We factor it using NMF and SVD, and compare the two approaches.
  - [Singular Value Decomposition](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Singular-Value-Decomposition-(SVD))
  - [Non-negative Matrix Factorization (NMF)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Non-negative-Matrix-Factorization-(NMF))
  - [Topic Frequency-Inverse Document Frequency (TF-IDF)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#TF-IDF)
  - [Stochastic Gradient Descent (SGD)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#NMF-from-scratch-in-numpy,-using-SGD)
  - [Intro to PyTorch](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#PyTorch)
  - [Truncated SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Truncated-SVD)
  
### [4. Randomized SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb) 
  - [Random Projections with word vectors](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-1:-Random-Projections-(with-word-vectors))
  - [Random SVD for Background Removal](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-2:-Random-SVD-for-Background-Removal)
  - [Timing Comparison](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Timing-Comparison)
  - [Math Details](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Math-Details)
  - [Random SVD for Topic Modeling](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-3:-Random-SVD-for-Topic-Modeling)

### [5. LU Factorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb)
 - [Gaussian Elimination](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Gaussian-Elimination)
 - [Stability of LU](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Stability)
  - [LU factorization with Pivoting](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#LU-factorization-with-Partial-Pivoting)
  - [History of Gaussian Elimination](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#History-of-Gaussian-Elimination)
  - [Block Matrix Multiplication](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Block-Matrices)

### [6. Compressed Sensing of CT Scans with Robust Regression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb)  
Compressed sensing is critical to allowing CT scans with lower radiation-- the image can be reconstructed with less data.  Here we will learn the technique and apply it to CT images.
  - [Broadcasting](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Broadcasting)
  - [Sparse matrices](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Sparse-Matrices-(in-Scipy))
  - [CT Scans and Compressed Sensing](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Today:-CT-scans)
  - [L1 and L2 regression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Regresssion)

### [7. Predicting Health Outcomes with Linear Regressions](https://github.com/fastai/numerical-linear-algebra-v2/blob/master/nbs/07-Health-Outcomes-with-Linear-Regression.ipynb) 
  - Linear regression in sklearn
  - Polynomial Features
  - Speeding up with Numba
  - Regularization and Noise

### [8. How to Implement Linear Regression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/08-How-to-Implement-Linear-Regression.ipynb)
  - How did Scikit Learn do it?
  - Naive solution
  - Normal equations and Cholesky factorization
  - QR factorization
  - SVD
  - Timing Comparison
  - Conditioning & Stability
  - Full vs Reduced Factorizations
  - Matrix Inversion is Unstable

### [9. PageRank with Eigen Decompositions](https://github.com/fastai/numerical-linear-algebra-v2/blob/master/nbs/09-PageRank-with-Eigen-Decompositions.ipynb)
We have applied SVD to topic modeling, background removal, and linear regression. SVD is intimately connected to the eigen decomposition, so we will now learn how to calculate eigenvalues for a large matrix.  We will use DBpedia data, a large dataset of Wikipedia links, because here the principal eigenvector gives the relative importance of different Wikipedia pages (this is the basic idea of Google's PageRank algorithm).  We will look at 3 different methods for calculating eigenvectors, of increasing complexity (and increasing usefulness!).
  - SVD  
  - DBpedia Dataset
  - Power Method
  - QR Algorithm
  - Two-phase approach to finding eigenvalues 
  - Arnoldi Iteration

### [10. Implementing QR Factorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/10-Implementing-QR-Factorization.ipynb)
  - Gram-Schmidt
  - Householder
  - Stability Examples

<hr>
