import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--scenario', type=int, default=1, help="simulation scenario")    
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--target_dataset', type=str, default='MNIST', help="name of dataset")
    parser.add_argument('--pre_trained_dataset', default='EMNIST', help='data on which the initial model has been pretrained, e.g. EMNIST, FMNIST')
    parser.add_argument('--dataset_split', type=str, default='balanced', help="type of split for EMNIST")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', default='True', help='aggregation over all clients')

    # required folders and files
    parser.add_argument('--data_root_dir', default='../data', help='data location')
    parser.add_argument('--model_root_dir', default='../model_weights', help='clustering model location')
    parser.add_argument('--results_root_dir', default='../results', help='results location')
    parser.add_argument('--log_root_dir', default='../logs', help='results location')
    #parser.add_argument('--ae_model_name', default='model-1606927012-epoch40-latent128', help='Autoencoder model name')
    #parser.add_argument('--pre_trained_dataset', default='EMNIST', help='data on which the initial model has been pretrained')

    # clustering options
    parser.add_argument('--clustering_method', default='umap_central', help='clustering method: single, local, perfect, umap_mo, umap, encoder, sequential_encoder, umap_central')
    parser.add_argument('--nr_of_clusters', default=5, help='number of clusters')
    parser.add_argument('--flag_with_overlap', default=False, help='clustering with overlapped labels')
    parser.add_argument('--flag_soft_clustering', default=False, help='clustering with soft thresholds')

    # AE model
    parser.add_argument('--ae_model_name', default='model-1607623811-epoch40-latent128', help='Autoencoder model name')
    parser.add_argument('--latent_dim', default=128, help='latent dimension')
    parser.add_argument('--manifold_dim', default=2, help='manifold dimension')
    parser.add_argument('--num_hiddens', default=2, help='number of hidden layers')
    parser.add_argument('--num_residual_hiddens', default=2, help='number of residual hidden layers')
    parser.add_argument('--num_residual_layers', default=2, help='number of residual layers in the stack')
    parser.add_argument('--nr_epochs_sequential_training', default=2, help='number of epochs for training the encoder')

    # third party results
    parser.add_argument('--multi_center', default=False, help='generate results for multi_center paper')

    # config file settings
    parser.add_argument('--weithed_evaluation', default=False, help='enable weithed accuracy evaluation')
    parser.add_argument('--iter_to_iter_results', type=int, default=1, help='generate results for each iteration')
    parser.add_argument('--config_root_dir', default='../configs', help='config file location')

    # table 3 config
    parser.add_argument('--change_dataset_flag', default=False, help='generating results for a scenario where the dataset is changed')
    parser.add_argument('--change_dataset_epoch', type=int, default=5, help='epcoh number where the dataset is changed')

    args = parser.parse_args()
    return args