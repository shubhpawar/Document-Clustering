# Docker Environment
Docker allows an easy execution of the provided scripts by creating a container that installs all the needed dependencies. This folder contains a Dockerfile that installs a Python 3.6 environment. The environment can be used to run and evaluate the provided PubMed abstract clustering algorithm.


## Setup
First, install Docker as described in the documentation: https://docs.docker.com/engine/installation/

Then, build, from the root folder of the repository, the docker container. Run:
```
docker build ./docker/ -t pubmed_clustering
```

This builds the Python 3.6 container and assigns the name *pubmed_clustering* to it.

To run the code, we must first start the container and mount the current folder $PWD into the container:
```
docker run -it -v "$PWD":/src pubmed_clustering bash
```

The command `-v "$PWD":/src` maps the current folder `$PWD` into the docker container at the position `/src`. Changes made on the host system as well as in the container are synchronized. We can change / add / delete files in the current folder and its subfolder and can access those files directly in the docker container.

Windows users can use the command `%cd%` instead of `$PWD` to get a path to current folder.

In this container, you can run execute the scripts as usual. For example, to cluster the abstracts, run:
```
python cluster_abstracts.py pmids_gold_set_unlabeled.txt
```
