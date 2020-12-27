# DIFF GRAPH U-NET: A DIFFERENTIAL GRAPH U-SHAPE NETWORK FOR GRAPH CLASSIFICATION
## Introduction
Regarding GCNs, there are mainly three tasks: node classification, link prediction, and graph classification. This project focuses on the classification of graphics through Differentiable Pooling (DIFFPOOL). DIFFPOOL is a state-of-the-art pool module that can learn the hierarchical expression of graphics and combine various end-to-end GCN structures. Hence, in order to research the changes of a graph after several DIFFPOOL modules respectively, the visualizations for pooling has been fulfilled via rebuilding the graph. Apart from DIFFPOOL, this project also explored the possibility to build an encoder-decoder architecture via building such a U-net-like method for graph data. A novel GCN architecture DIFF Graph U-Net is proposed. 

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