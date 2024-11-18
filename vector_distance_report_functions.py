#######################################
# Threshold Boolean Distance Functions
#######################################

from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_distance(embedding1, embedding2, boolean=False, threshold=0.6):
    """
    Cosine Similarity: This is a common method for measuring the similarity
    between two vectors. It measures the cosine of the angle between
    two vectors and the result is a value between -1 and 1.
    A value of 1 means the vectors are identical,
    0 means they are orthogonal (or completely dissimilar),
    and -1 means they are diametrically opposed.

    if not surprisingly, this looks solid: gold standard?
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = cosine_similarity([embedding1], [embedding2])

    similarity = similarity[0][0]


    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile

from scipy.spatial.distance import euclidean

def euclidean_distance(embedding1, embedding2, boolean=False, threshold=0.5):
    """
    Euclidean Distance: This is another common method for measuring
     the similarity between two vectors.
     It calculates the straight-line distance between two points in a space.
     The smaller the distance, the more similar the vectors.
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + euclidean(embedding1, embedding2))


    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


import numpy as np

def normalized_dot_product(embedding1, embedding2, boolean=False, threshold=0.6):
    """
    Dot Product: This is a simple method that calculates
    the sum of the products of the corresponding entries of the
    two sequences of numbers. If the vectors are normalized,
    the dot product is equal to the cosine similarity.

    0.5 ok? seems good
    """
    # Assuming embedding1 and embedding2 are your embeddings
    dot_product = np.dot(embedding1, embedding2)
    normalized_dot_product = dot_product / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    similarity = normalized_dot_product


    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile

from scipy.spatial.distance import cityblock

def manhattan_distance(embedding1, embedding2, boolean=False, threshold=0.0024):
    """
    Manhattan Distance: This is a measure of the distance between
    two vectors in a grid-based system.
    It calculates the sum of the absolute differences of their coordinates.
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + cityblock(embedding1, embedding2))


    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.stats import pearsonr

def pearson_correlation(embedding1, embedding2, boolean=False, threshold=0.6):
    """
    Pearson Correlation: This is a measure of the linear correlation
    between two vectors. It ranges from -1 (perfectly negatively correlated)
     to 1 (perfectly positively correlated).

    maybe decent around 0.6?
    """
    
    # Assuming embedding1 and embedding2 are your embeddings
    similarity, _ = pearsonr(embedding1, embedding2)

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.stats import spearmanr

def spearmans_rank_correlation(embedding1, embedding2, boolean=False, threshold=0.6):
    """
    Spearman's Rank Correlation: This is a non-parametric
     measure of the monotonicity of the relationship between
     two datasets. Unlike the Pearson correlation, the Spearman
      correlation does not assume that the relationship between
       the two variables is linear.

    more strict measure?
    """
    
    # Assuming embedding1 and embedding2 are your embeddings
    similarity, _ = spearmanr(embedding1, embedding2)


    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile

from scipy.stats import kendalltau
def kendalls_rank_correlation(embedding1, embedding2, boolean=False, threshold=0.7):
    
    """
    Kendall's Rank Correlation: This is another non-parametric
    measure of the ordinal association between two variables.
    It is a measure of the correspondence between two rankings.

    0.3 may match the subject generally
    0.5 may most closely match meaning
    """
    
    # Assuming embedding1 and embedding2 are your embeddings
    similarity, _ = kendalltau(embedding1, embedding2)

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.spatial.distance import minkowski


def minkowski_distance(embedding1, embedding2, boolean=False, threshold=0.055):
    """
    Minkowski Distance: This is a generalization of
    both the Euclidean distance and the Manhattan distance.
    It is defined as the p-th root of the sum of the p-th powers
    of the differences of the coordinates.
    When p=1, this is the Manhattan distance,
    and when p=2, this is the Euclidean distance.
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + minkowski(embedding1, embedding2, p=2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.spatial.distance import chebyshev
def chebyshev_distance(embedding1, embedding2, boolean=False, threshold=0.4):
    """
    Chebyshev Distance: This is a measure of the distance between
    two vectors in a vector space.
    It is the maximum of the absolute differences of their coordinates.
    """
    
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + chebyshev(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


import numpy as np
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def mahalanobis_distance(embedding1, embedding2, boolean=False, threshold=0.415):
    """Mahalanobis Distance: This is a measure of the distance between 
    a point P and a distribution D, introduced by P. C. Mahalanobis in 1936.
    It is a multivariate generalization of the Euclidean distance.
    It is based on correlations between dimensions of the data, 
    and thus takes into account the structure of the data.
    """

    # Assuming embedding1 and embedding2 are your vectors
    data = np.array([embedding1, embedding2])

    # Calculate the covariance matrix with a small regularization term
    cov = np.cov(data, rowvar=False) + np.eye(data.shape[1])# * 1e-6

    # Calculate the Mahalanobis distance
    distance = mahalanobis(embedding1, embedding2, inv(cov))

    # Calculate the similarity score
    similarity = 1 / (1 + distance)

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile



from scipy.spatial.distance import braycurtis
def bray_curtis_distance_dissimilarity(embedding1, embedding2, boolean=False, threshold=0.75):
    """Bray-Curtis Distance: This is a measure of dissimilarity
    between two vectors. It is used in ecology to compare species
    composition in different samples. It is defined as the sum of
    the absolute differences between the vectors, divided by the sum of their sums.

    0.75 is maybe a stricker-yes

    but total no is still .6+
    """
    
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + braycurtis(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.spatial.distance import canberra
def canberra_distance(embedding1, embedding2, boolean=False, threshold=0.002):
    """
    dissimilarity
    Canberra Distance: This is a measure of the dissimilarity
    between two vectors. It is defined as the sum of the absolute
    differences between the vectors, divided by the sum of their absolute values.
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + canberra(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile



from scipy.stats import pearsonr
def correlation_distance_dissimilarity_measure(embedding1, embedding2, boolean=False, threshold=0.7):
    """
    dissimilarity
    Correlation Distance: This is a measure of the dissimilarity
    between two vectors. It is defined as 1 - the absolute value of
    the Pearson correlation coefficient between the vectors.

    even no is hight... maybe .7 ok?
    """
    # Assuming embedding1 and embedding2 are your embeddings
    correlation, _ = pearsonr(embedding1, embedding2)
    similarity = 1 / (1 + (1 - abs(correlation)))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile



from scipy.spatial.distance import sqeuclidean
def squared_euclidean_distance_dissimilarity_measure(embedding1, embedding2, boolean=False, threshold=0.005):
    """
    dissimilarity
    Squared Euclidean Distance: This is a measure of the dissimilarity
    between two vectors. It is defined as the sum of the squared differences
    between the vectors. It is similar to the Euclidean distance,
    but it does not take the square root, which can make it faster to compute.
    """
    # Assuming embedding1 and embedding2 are your embeddings
    similarity = 1 / (1 + sqeuclidean(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile

from scipy.spatial.distance import hamming
def hamming_distance_dissimilarity_measure(embedding1, embedding2):
    """
    Hamming Distance: This is a measure of the minimum number
    of substitutions required to change one vector into the other.
    It is used in information theory to measure the difference between
    two binary vectors.
    """
    # Assuming embedding1 and embedding2 are your binary vectors
    similarity = 1 / (1 + hamming(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile


from scipy.stats import wasserstein_distance
def total_variation_distance_dissimilarity_measure(embedding1, embedding2, boolean=False, threshold=0.97):
    """
    dissimilarity
    Total Variation Distance: This is a measure of the dissimilarity
    between two probability distributions.
    It is defined as half the sum of the absolute differences
    between the corresponding probabilities in the two distributions.

    all scores high, maybe .97 is strict enough?
    """
    # Assuming embedding1 and embedding2 are your probability distributions
    similarity = 1 / (1 + wasserstein_distance(embedding1, embedding2))

    threshold_difference = similarity - threshold

    boolean_result = None

    if similarity < threshold:
        boolean_result = False

    else:
        boolean_result = True

    profile = {
        'boolean': boolean_result,
        'threshold': threshold,
        'threshold_difference': threshold_difference,
        'similarity_measure': similarity,
    }

    return profile

