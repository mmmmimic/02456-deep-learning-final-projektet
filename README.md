# DIFF GRAPH U-NET: A DIFFERENTIABLE U-SHAPE NETWORK FOR GRAPH CLASSIFICATION
## Introduction
Recently, graph convolutional networks (GCNs) have been demonstrated efficient in learning graph representations. Regarding the down-sampling and up-sampling of non-Euclidean data, most existing methods are flat and lack robustness. We visualize the process of a state-of-the-art work DiffPool, and develop a novel differentiable module for upsampling called DiffUnpool. DiffPool and DiffUnpool learn soft cluster assignment for nodes via GCNs and multi-layer perceptrons respectively. To address the graph classification problem, based on DiffPool and DiffUnpool, we further propose an end-to-end encoder-decoder architecture, diff graph U-Net. Different from other U-shape models before, diff graph U-Net learns node embeddings hierarchically, and collect global features in residual fashion. Our experimental results show that our model yields an overall improvement of accuracy on 4 different data sets, compared with previous methods.  

## Author
Manxi Lin s192230

Mengge Hu s192113

Guangya Shen s200104

## Data set
https://drive.google.com/file/d/1nTM9c4HgIeb6iFauLQABuGjqDGpc43iv/view?usp=sharing

Unzip it in here

## Check our main result
- See `main.ipynb`
- Proof of our result: `./screenshots`

## Our contribution
- Revision in `encoders.py` and `train.py` to implement DIFF Graph U-Net
- Implement `sample_data.py` to sample data sets
- Display our main result in a jupyter notebook `main.ipynb`

## DIFFPOOL
Repo Link https://github.com/RexYing/diffpool

## DIFFPOOL Introduction (from the origional repo)
Recently, graph neural networks (GNNs) have revolutionized the field of graph
representation learning through effectively learned node embeddings, and achieved
state-of-the-art results in tasks such as node classification and link prediction.
However, current GNN methods are inherently flat and do not learn hierarchical
representations of graphs—a limitation that is especially problematic for the task
of graph classification, where the goal is to predict the label associated with an
entire graph. Here we propose DIFFPOOL, a differentiable graph pooling module
that can generate hierarchical representations of graphs and can be combined with
various graph neural network architectures in an end-to-end fashion. DIFFPOOL
learns a differentiable soft cluster assignment for nodes at each layer of a deep
GNN, mapping nodes to a set of clusters, which then form the coarsened input
for the next GNN layer. Our experimental results show that combining existing
GNN methods with DIFFPOOL yields an average improvement of 5–10% accuracy
on graph classification benchmarks, compared to all existing pooling approaches,
achieving a new state-of-the-art on four out of five benchmark data sets.


Paper link: https://arxiv.org/pdf/1806.08804.pdf
