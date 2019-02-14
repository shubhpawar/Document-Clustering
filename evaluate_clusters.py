from cluster_abstracts import PubMedClustering
from sklearn import metrics
from sklearn import preprocessing
import argparse


def perform_clustering(
        filename,
        n_components,
        num_clusters,
        f_op,
        num_words,
        f_store):
    """ Performs clustering on abstracts. """
    clusterer = PubMedClustering(
        filename,
        n_components,
        num_clusters,
        num_words,
        f_store)
    clusterer.cluster_abstracts()
    clusterer.get_top_terms()

    if f_op:
        clusterer.store_output()

    return clusterer


def get_true_labels(filename):
    """ Extracts gold cluster labels from the input file. """
    with open('data/' + filename, 'r') as f:
        lines = f.readlines()

    # Checking if gold labels exist in the input file
    if len(lines[0].split()) > 1:
        cluster_labels = [' '.join(line.split()[1:]) for line in lines]
        # Encoding the string labels to numeric labels
        le = preprocessing.LabelEncoder()
        labels_true = le.fit_transform(cluster_labels)
    else:
        labels_true = None

    return labels_true


def evaluate_clusters(model, labels_true, f1, f2, f3, dist_metric):
    """ Computes various evaluation metrics to assess clustering. """
    labels_pred = model.clusters

    print('Computing evaluation metric(s)...')
    if labels_true is not None:
        if f1:
            print(
                "Adjusted Rand index: %0.3f" %
                metrics.adjusted_rand_score(
                    labels_true,
                    labels_pred))
        if f2:
            print(
                "Homogeneity: %0.3f" %
                metrics.homogeneity_score(
                    labels_true,
                    labels_pred))
            print(
                "Completeness: %0.3f" %
                metrics.completeness_score(
                    labels_true,
                    labels_pred))
            print(
                "V-measure: %0.3f" %
                metrics.v_measure_score(
                    labels_true,
                    labels_pred))
    else:
        print("Gold labels are not provided.")

    if f3:
        print(
            "Silhouette Coefficient: %0.3f" %
            metrics.silhouette_score(
                model.X,
                labels_pred,
                metric=dist_metric))


def parse_arguments():
    """ Parses command-line arguments. """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'filename',
        type=str,
        help='name of the file containing PubMed article IDs')
    parser.add_argument(
        '--lsa',
        dest='n_components',
        type=int,
        default=100,
        help='dimensionality of latent semantic analysis output')
    parser.add_argument(
        '--num_clusters',
        dest='num_clusters',
        type=int,
        default=6,
        help='number of clusters to form')
    parser.add_argument(
        '--ari',
        action='store_true',
        help='adjusted rand index (ARI)')
    parser.add_argument(
        '--hcv',
        action='store_true',
        help='homogeneity, completeness and v-measure')
    parser.add_argument(
        '--sc',
        action='store_true',
        help='mean silhouette coefficient')
    parser.add_argument(
        '--dist',
        dest='dist_metric',
        type=str,
        default='euclidean',
        help='distance metric for silhouette coefficient')
    parser.add_argument(
        '--store_output',
        action='store_true',
        help='store output clusters in a file')
    parser.add_argument(
        '--num_words',
        dest='num_words',
        type=int,
        default=5,
        help='number of top words per cluster')
    parser.add_argument(
        '--store_model',
        action='store_true',
        help='persist the model on disk')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    model = perform_clustering(
        args.filename,
        args.n_components,
        args.num_clusters,
        args.store_output,
        args.num_words,
        args.store_model)
    labels_true = get_true_labels(args.filename)
    evaluate_clusters(
        model,
        labels_true,
        args.ari,
        args.hcv,
        args.sc,
        args.dist_metric)
