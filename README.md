# Clustering PubMed Abstracts using K-Means

This repository contains an implementation of a clustering approach for grouping PubMed abstracts that uses K-Means algorithm.

The implementation contains two scripts: cluster_abstracts.py and evaluate_clusters.py.

* **cluster_abstracts.py**: This script, given a list of PMIDs, retrieves the corresponding abstracts, clusters them using k-means and outputs the predicted clusters in a text file. PMID is the unique identifier number used in PubMed. They are assigned to each article record when it enters the PubMed system.

* **evaluate_clusters.py**: This script, given a list of PMIDs (with or without gold cluster labels), evaluates the clustering performance using various metrics.

# Setup
In order to run the code, I recommend Python 3.6.


## Setup with virtual environment (Python 3)

Setup a Python virtual environment (optional):
```
python3 -m venv ~/.virtualenvs/env_name
source ~/.virtualenvs/env_name/bin/activate
```

Install the requirements:
```
pip3 install -r requirements.txt
```

## Setup with docker
See the [docker-folder](docker/) for more information on how to run these scripts in a docker container.


# Clustering the abstracts
To cluster the abstracts corresponding to the PMIDs in a text file, run the following command:

```
python3 cluster_abstracts.py input_filename.txt
```

This script will read the text file `input_filename.txt`. The text is split to extract PMIDs and those are used to extract corresponding PubMed abstracts using BioPython. These abstracts are tokenized and stemmed using NLTK and transformed into tf-idf feature matrix using Scikit-learn. LSA is used to reduce dimensionality and K-Means to cluster the samples.

You can use the following command-line parameters while executing this script:

- **filename**:  The name of the file containing PubMed article IDs. This is a positional argument.
- **lsa**: The dimensionality of the latent semantic analysis output. This is a conditional argument. Default: 100
- **num_clusters**: The number of clusters to form. This is a conditional argument. Default: 6
- **num_words**: The number of top words per cluster. This is a conditional argument. Default: 5
- **store_model**: The flag to persist the model on disk. This is a conditional argument. Default: False

# Evaluating the clustering performance
To evaluate the results of clustering, run the following command:

```
python3 evaluate_clusters.py input_filename.txt
```

This script uses cluster_abstracts.py script to obtain the clustering results. It also extracts the gold cluster labels if they are present in the text file `input_filename.txt`. It computes and prints the following evaluation metrics to assess the clustering performance: Adjusted Rand Index, Homogeneity, Completeness, V-Measure and Silhouette Coefficient.

You can use the following command-line parameters while executing this script:

- **filename**:  The name of the file containing PubMed article IDs (and corresponding cluster labels). This is a positional argument.
- **lsa**: The dimensionality of the latent semantic analysis output. This is a conditional argument. Default: 100
- **num_clusters**: The number of clusters to form. This is a conditional argument. Default: 6
- **ari**: The flag to compute Adjusted Rand Index (ARI). This is a conditional argument. Default: False
- **hcv**: The flag to compute Homogeneity, Completeness and V-Measure. This is a conditional argument. Default: False
- **sc**: The flag to compute mean Silhouette Coefficient. This is a conditional argument. Default: False
- **dist**: The distance metric used for computing silhouette coefficient. This is a conditional argument. Default: False
- **store_output**: The flag to store output clusters in a file. This is a conditional argument. Default: False
- **num_words**: The number of top words per cluster. This is a conditional argument. Default: 5
- **store_model**: The flag to persist the model on disk. This is a conditional argument. Default: False
