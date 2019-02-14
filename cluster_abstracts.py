import nltk
import re
import pickle
import argparse
import pandas as pd
from Bio import Entrez
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer


class PubMedClustering:
    """PubMed Abstract Clusterer.

    Attributes:
        filename: Input filename containing PMIDs.
        n_components: Dimensionality of LSA output.
        num_clusters: Number of clusters to form.
    """

    def __init__(
            self,
            filename,
            n_components=100,
            num_clusters=6,
            num_words=5,
            f_store=False):
        """ Sets up PubMed Clusterer. """
        self.filename = filename
        self.n_components = n_components
        self.n_clusters = num_clusters
        self.n_words = num_words
        self.f_store = f_store

        # Getting the NLTK English language stopword list.
        self.stopwords = nltk.corpus.stopwords.words('english')

        # Stemming the stopwords for consistency.
        stemmer = SnowballStemmer("english")
        self.stemmed_stopwords = [
            stemmer.stem(w) for w in self.stopwords] + [
            "'d",
            'could',
            'might',
            'must',
            "n't",
            'need',
            'r',
            'sha',
            'v',
            'wo',
            'would']

    def retrieve_abstracts(self):
        """ Retrieves PubMed abstracts from the NCBI Entrez database. """
        with open('data/' + self.filename, 'r') as f:
            lines = f.readlines()

        self.pmids = [line.rsplit()[0] for line in lines]

        Entrez.email = 'spawar3@uncc.edu'

        # Fetching PubMed records from the NCBI Entrez DB.
        print('Retrieving PubMed abstracts...')
        handle = Entrez.efetch(
            db="pubmed",
            id=','.join(self.pmids),
            rettype="xml",
            retmode="text")
        records = Entrez.read(handle)

        abstracts = []

        # Extracting abstract text from PubMed articles.
        for article in records['PubmedArticle']:
            if 'Abstract' in article['MedlineCitation']['Article'].keys():
                abstract = article['MedlineCitation']['Article']['Abstract']
                abstract_text = abstract['AbstractText'][0]
                abstracts.append(abstract_text)
            else:
                # Using article title as a proxy for abstract when abstract is
                # missing.
                abstracts.append(
                    article['MedlineCitation']['Article']['ArticleTitle'])

        return abstracts

    def tokenize(self, text, stem=True):
        """ Preprocesses the text data. """
        # Tokenizing the abstracts first by sentence, then by word.
        tokens = [word for sent in nltk.sent_tokenize(
            text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # Filtering out any tokens not containing letters (e.g. numeric tokens,
        # raw punctuation, etc.)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)

        if stem:
            # Stemming the tokens.
            stemmer = SnowballStemmer("english")
            tokens = [stemmer.stem(t) for t in filtered_tokens]
            return tokens
        else:
            return filtered_tokens

    def cluster_abstracts(self):
        """ Clusters abstracts using K-Means. """
        self.abstracts = self.retrieve_abstracts()

        # Converting abstracts to a matrix of TF-IDF features.
        print('Extracting tf-idf features...')
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stemmed_stopwords,
            use_idf=True,
            tokenizer=self.tokenize,
            ngram_range=(1, 3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.abstracts)

        self.terms = tfidf_vectorizer.get_feature_names()

        # Performing dimensionality reduction using LSA and normalizing the
        # results.
        print('Performing dimensionality reduction...')
        svd = TruncatedSVD(self.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        self.X = lsa.fit_transform(tfidf_matrix)

        # Applying K-Means to obtain clusters
        print('Clustering abstracts...')
        self.km = KMeans(n_clusters=self.n_clusters, n_jobs=-1, n_init=10)
        self.km.fit(self.X)

        if self.f_store:
            print('Storing model to disk...')
            with open('models/clusterer_model.pkl', 'wb') as f:
                pickle.dump(self.km, f)

        self.clusters = self.km.labels_.tolist()

        self.original_space_centroids = svd.inverse_transform(
            self.km.cluster_centers_)

    def get_top_terms(self):
        """ Finds top terms or "composition" of clusters. """
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in self.abstracts:
            allwords_stemmed = self.tokenize(i, stem=True)
            totalvocab_stemmed.extend(allwords_stemmed)

            allwords_tokenized = self.tokenize(i, stem=False)
            totalvocab_tokenized.extend(allwords_tokenized)

        # Creating a mapping from stems to their original forms/tokens
        self.stem_to_token = {}
        for i, term in enumerate(totalvocab_stemmed):
            if term not in self.stem_to_token.keys():
                self.stem_to_token[term] = totalvocab_tokenized[i]

        # Sorting to identify which are the top words that are nearest to the
        # cluster centroid (to extract cluster "composition")
        self.order_centroids = self.original_space_centroids.argsort()[:, ::-1]

        self.cluster_top_terms = {}
        # Getting top terms per cluster
        for i in range(self.n_clusters):
            cluster_terms = []
            for ind in self.order_centroids[i, :self.n_words]:
                term = ''
                for token in self.terms[ind].split():
                    term += self.stem_to_token[token] + ' '
                cluster_terms.append(term)
            self.cluster_top_terms[str(i)] = cluster_terms

    def store_output(self):
        """ Stores the clustering output to a text file. """
        self.cluster_compositions = [', '.join(
            self.cluster_top_terms[str(cluster)]) for cluster in self.clusters]

        self.pubmed_articles = {
            'pmid': self.pmids,
            'abstract': self.abstracts,
            'cluster': self.clusters,
            'cluster_composition': self.cluster_compositions}

        self.pm_article_df = pd.DataFrame(self.pubmed_articles)

        print('Storing clustering output...')
        self.pm_article_df.to_csv(
            'output/output_clusters.txt',
            columns=[
                'pmid',
                'cluster_composition'],
            header=False,
            index=False,
            sep=' ')


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

    clusterer = PubMedClustering(
        args.filename,
        args.n_components,
        args.num_clusters,
        args.num_words,
        args.store_model)
    clusterer.cluster_abstracts()
    clusterer.get_top_terms()
    clusterer.store_output()
