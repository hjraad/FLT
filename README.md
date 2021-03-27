## Federated Learning with Taskonomy (FLT)
This is the official repository for [Federated Learning with Taskonomy (FLT) for Non-IID Data]() (Hadi Jamali-Rad, Mohammad Abdizadeh, Attila Szabó)

### Abstract
Classical federated learning approaches incur significant performance degradation in the presence of non-IID client data. A possible direction to address this issue is forming clusters of clients with roughly IID data. Most solutions following this direction are iterative, and relatively slow and prone to convergence issues in discovering underlying cluster structure. 

We introduce federated learning with taskonomy (FLT) that generalizes this direction by learning the task-relatedness between clients for more efficient federated aggregation of heterogeneous data. In a one-off process, the server provides the clients with a pretrained encoder to compress their data into a latent representation, and transmit the signature of their data back to the server. The server then learns the task-relatedness among clients via manifold learning, and performs a generalization of federated averaging. FLT can flexibly handle generic client relatedness as well as decomposing it into (disjoint) cluster formation. 
 
We demonstrate that FLT not only outperforms the existing state-of-the-art baselines but also offers improved fairness across clients.

### Architecture
<img src="Figures/architecture.png" width="500" >

We consider three abstraction levels: 
- **data level**,
- **encoder level**, where a contractive latent space representation of client data is extracted in an unsupervised fashion
- **manifold approximation level** with UMAP

The encoder is provided by the server to the clients. This allows them to apply one-shot contractive encoding on their local data, followed by k-means on the outcome and return the results to the server. At server side, UMAP is applied to approximate the arriving clients embeddings. 
This is followed by applying a distance threshold to determine client dependencies and form an adjacency matrix or a client (task) relatedness graph. In case forming disjoint clusters is of interest, we then use hierarchical clustering [1] to efficiently reorder the adjacency matrix (or corresponding client relatedness graph) into disjoint clusters. We show a) the adjacency matrix and b) the corresponding client relatedness graph (both reordered on the right) in the following figure:

<img src="Figures/graph_adjacency_2.png" width="400" >

### Requirements
1. Go to the root directory ```cd FLT```
2. Create an environment ```conda create -n flt python=3.7```
3. Activate the environment ```conda activate flt```
4. Install the requirements ```pip install -r requirements.txt```

### Run experiments
The main script should be run from the right directory as follows
```cd base_fed_learning```
```python main_fed.py```

### Contact
Corresponding author: Hadi Jamali-Rad (h.jamali.rad {at} gmail {dot} com, h.jamalirad {at} tudelft {dot} nl)

**References**
[1] Modern hierarchical, agglomerative clustering algorithms, D. Müllner, 2011.
<!-- [1] An efficient framework for clustered federated learning, A. Gosh, J. Chung, D. Yin, and K. Ramchandran, 2020. -->
<!-- [2] Multi-center federated learning, M. Xie, G. Long, T. Shen, T. Zhou, X. Wang, and J. Jiang, 2020. -->
<!-- [3] Heterogeneity for the Win: Communication-Efficient Federated Clustering, D. K. Dennis and V. Smith, 2020.  -->