from time import perf_counter

from sklearn import random_projection
from sklearn.decomposition import IncrementalPCA

#IPCA from scikit-learn. We choose batch_size empirically
def ipca(data,components):
    start = perf_counter()
    ipca = IncrementalPCA(n_components=components, batch_size=1000)
    X = ipca.fit_transform(data)
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish dimension reduction with IPCA.')
    return X

#Sparse random projection
def sparse_random_projection(data):
    start = perf_counter()
    transformer = random_projection.SparseRandomProjection(random_state=104)
    X = transformer.fit_transform(data)
    end = perf_counter()
    #String contains: It took {end-start} second(s) to finish
    #dimension reduction with Sparse Random Projection.
    print(f'It took {end-start} second(s) to finish dimension reduction with Sparse Random Projection.')
    return X