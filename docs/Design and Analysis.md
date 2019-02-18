# PubMed Clustering Problem

The task is to cluster PubMed abstracts into groups, given their corresponding PMIDs, that would ideally match the search query results.

My approach is described below:

* **Step 1**: Document Retrieval

Given a list of PMIDs, I use the Bio.Entrez module to access the NCBI Entrez database and retrieve the corresponding PubMed abstracts. I also noticed that some PMIDs don't have associated abstract text. In that case, I use the article title as a proxy for the abstract text.

Note: Initially, for some reason, I was not able to access NCBI websites/API because of an "Access Denied" error message. I found out that it's a common problem - https://libertyscientist.wordpress.com/2012/01/26/access-denied-the-tragedy-of-the-commons-strikes-again/. However, I was able to avoid this problem by using my university's VPN.

* **Step 2**: Preprocessing Text

- **Tokenization**: It is a process of parsing text data into smaller units (tokens) such as words and phrases. I use unigrams, bigrams and trigrams as many disease names are multi-word phrases.

- **Stop Word and Punctuation Removal**: Some tokens have less importance than others. For example, common words such as 'an' and punctuation such as '!' don't reveal the essential characteristics of a text.

- **Stemming**: Various tokens might carry similar information. For example, 'going' and 'went' have the same information content. Hence, to reduce the redundant information, stemming reduces inflected (or sometimes derived) words to their word stem. I use the NLTK Snowball stemmer. Lemmatization could also be used which is a process of determining the lemma of a word based on its intended meaning.

* **Step 3**: TF-IDF Feature Extraction

Tf-idf is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. I use sklearn TfidfVectorizer. It uses a in-memory vocabulary (a python dict) to map the most frequent words to features indices and hence compute a word occurrence frequency (sparse) matrix. The word frequencies are then reweighted using the Inverse Document Frequency (IDF) vector collected feature-wise over the corpus.

* **Step 4**: Dimensionality Reduction

After extracting tf-idf features, I perform dimensionality reduction on feature vectors using truncated singular value decomposition (SVD), also known as latent semantic analysis (LSA) in this context. It also helps in discovering latent patterns in the data.

* **Step 5**: Clustering

I chose **K-Means** algorithm to cluster the abstracts. The K-Means algorithm clusters data by trying to separate samples in k groups of equal variance, minimizing a criterion known as the inertia (aka distortion) or within-cluster sum-of-squares. The algorithm requires the number of clusters (k) to be specified.

The algorithms starts with initial estimates for the k centroids, which can either be randomly generated or randomly selected from the data set. The algorithm then iterates between two steps until convergence:

1. Cluster Assignment: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
2. Move Centroids: The centroids are recomputed by taking the mean of all data points assigned to that centroid's cluster.

**Euclidean similarity metric**: Euclidean distance is the straight-line distance between two points in Euclidean space.

K-Means is susceptible to being stuck in local optima. Hence, it might take multiple runs of the algorithm to reach a global optimum.

The average complexity is given by O(nkdi) where n is the number of d-dimensional vectors, k the number of clusters and i the number of iterations needed until convergence.

* **Step 6**: Evaluation

I implemented five metrics to evaluate the clustering performance:

- **Adjusted Rand Index**: Given the gold cluster assignments and clustering algorithm assignments, it is a function that measures the similarity of the two assignments, ignoring permutations and with chance normalization.

Bounded range [-1, 1]: negative values are bad (independent labelings), similar clusterings have a positive ARI, 1.0 is the perfect match score.

- **Homogeneity, Completeness and V-Measure**: They require knowledge of the ground truth class assignments.

Homogeneity: each cluster contains only members of a single class.
Completeness: all members of a given class are assigned to the same cluster.
V-measure: harmonic mean of homogeneity and completeness.

Bounded range [0, 1]: 0.0 is as bad as it can be, 1.0 is a perfect score.

- **Silhouette Coefficient**: It is a metric where a higher score related to a model with better defined clusters. It does not require ground truth labels.

The Silhouette Coefficient is defined for each sample and is composed of two scores:

  a: The mean distance between a sample and all other points in the same class.
  b: The mean distance between a sample and all other points in the next nearest cluster.

The Silhouette Coefficient s for a single sample is then given as:
  s = (b - a) / max(a, b)

I use the mean of the Silhouette Coefficient scores for all samples to evaluate clustering.

Bounded range [-1, 1]: -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.

## Performance on the Gold Set

As we already know the ideal amount of clusters for the gold set i.e. 6, I only focused on getting better scores for the evaluation metrics. As K-means often converges to a local optimum, I executed the algorithm multiple times to get better clusters. In general, my approach performed pretty well on the gold set. The clustering performance degrades for clusters with very few samples (e.g. 'spinal muscular atrophy' cluster).

The file 'output_clusters_gold_labeled.txt' contain the best clustering results I obtained on the gold set. Following were the evaluation scores for the same:

Adjusted Rand index: 0.917
Homogeneity: 0.905
Completeness: 0.918
V-measure: 0.912
Silhouette Coefficient: 0.038

## Performance on the Test Set

As we do not have gold cluster labels for the test set, I had to rely on the mean silhouette coefficient score to choose the best model. After testing out different values of number of clusters and looking at the clustering results for distinct, well-defined clusters, the best clustering results I obtained are in the file 'output_clusters_test_unlabeled.txt'. Associated silhouette score is 0.046 (with 'cosine' distance metric).

Looking at the output, we can see that the PubMed records in the test set are about the following six topics:

1. Noonan syndrome
2. Marfan syndrome
3. Lynch syndrome
4. Turner syndrome
5. CdLS
6. lung adenocarcinoma

The output cluster compositions could be more well-defined if biomedical stop words (such as 'patient', 'report', etc.) are removed.
