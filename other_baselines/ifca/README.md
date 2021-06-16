## Federated Learning with Taskonomy (FLT)
This is a fork of the official codebase for [ifca](https://github.com/jichan3751/ifca). The purpose of this fork is to use it as a baseline for FLT.

### Experiments
For the sake of a fair comparison, we run 500 rounds of initialization (with FedAvg) followed by another 1000 rounds of IFCA itself, a total of 1500 rounds for both MLP and CNN. We use the first 200 users of LEAF for all models. Please find more details in Scenario 3.

### Requirements
install requirements: 
```
cd FLT/other_baselines/ifca/femnist/
pip install -r requirements.txt
```

Copy data to ```FLT/other_baselines/ifca/femnist/fedavg_pretrain/data/femnist/data/{train,test}```folders
**Note**: To reproduce the results,  use ```all_data_{0,1}_niid_0_keep_0_{train,test}_9.json``` data files in the respective folder

### Run the experiments
```
cd FLT/other_baselines/ifca/femnist/
bash run_all.sh
```
**Note**: (by default the experiments are run with cnn models, you can change it to mlp ```FLT/other_baselines/ifca/femnist/fedavg_pretrain/run_fedavg_pretrain.sh and FLT/other_baselines/ifca/femnist/ifca/run_ifca.sh```)

# ifca
Codebase for [An Efficient Framework for Clustered Federated Learning](https://arxiv.org/abs/2006.04088).
