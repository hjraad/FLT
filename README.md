# Federated Learning with Taskonomy (FLT)
This is the official repository for [Federated Learning with Taskonomy (FLT) for Non-IID Data]() (Hadi Jamali-Rad, Mohammad Abdizadeh, Attila Szabó)

## Abstract
> Classical federated learning approaches incur significant performance degradation in the presence of non-IID client data. A possible direction to address this issue is forming clusters of clients with roughly IID data. Most solutions following this direction are iterative, and relatively slow and prone to convergence issues in discovering underlying cluster structure. 
> 
> We introduce federated learning with taskonomy (FLT) that generalizes this direction by learning the task-relatedness between clients for more efficient federated aggregation of heterogeneous data. In a one-off process, the server provides the clients with a pretrained encoder to compress their data into a latent representation, and transmit the signature of their data back to the server. The server then learns the task-relatedness among clients via manifold learning, and performs a generalization of federated averaging. FLT can flexibly handle generic client relatedness as well as decomposing it into (disjoint) cluster formation. 
> 
> We demonstrate that FLT not only outperforms the existing state-of-the-art baselines but also offers improved fairness across clients.

## Architecture
<img src="Figures/architecture.png" width="500" >

## Forming clusters with hierarchical clustering
<img src="Figures/graph_adjacency_2.png" width="300" >

## Requirements
Go to the root directory ```cd FLT```
Create an environment ```conda create -n flt python=3.7```
Activate the environment ```conda activate flt```
Install the requirements ```pip install -r requirements.txt```

## Run experiments
The main script should be run from the right directory as follows
```cd base_fed_learning```
```python main_fed.py```

## Contact
Corresponding author: dr. Hadi Jamali-Rad (h.jamali.rad {at} gmail {dot} com, h.jamalirad {at} tudelft {dot} nl)