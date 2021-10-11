import numpy as np
import matplotlib as plt
import scipy.optimize

np.random.seed(123)

nr_clients = 2400
nr_clusters = 10
nr_samples = 112800
nr_classes = 47

def func(x, y, C, K, F):
    # y: idx of user
    # C: constant not relevant
    # x: const. to get out of the equation
    # F: starting point (as first user should have 10)
    # K: total to reach (max for the cluster)
    return np.sum(F + np.exp(C*x*(y**2)), axis=0) - K

def sampling_support_EMNIST(nr_clients, nr_clusters, nr_classes, nr_samples, plot:bool):
    
    # Assgin the labels acc. cluster def.
    # Assign uncorrelated labells into clusters 

    # Assigning classes to clusters
    # it is robust against nr_clusters
    if nr_classes % nr_clusters > 0:
        (nr_classes_a, nr_classes_b) = nr_classes // (nr_clusters- 1), nr_classes % (nr_clusters - 1)
    else:
        (nr_classes_a, nr_classes_b) = nr_classes // (nr_clusters), nr_classes % (nr_clusters)
    
    (len_a, len_b) = np.floor(nr_samples/nr_classes * nr_classes_a), np.floor(nr_samples/nr_classes*nr_classes_b)
    
    CC = 1
    nr_clients_per_cluster = nr_clients // nr_clusters
    
    KK = int(len_a)
    FF = np.floor(len_a/2/nr_clients_per_cluster)
    yy = np.arange(1, nr_clients_per_cluster + 1)
    x = scipy.optimize.fsolve(func, x0=0, args=(yy[:, None], CC, KK, FF))
    li_a = [int(np.floor(FF + np.exp(CC*x*(i**2)))) for i in yy] # assigning labels to users based on pow law
    arr_a = np.array(li_a)
    arr_a[0] += KK - np.sum(arr_a) # the rest goes to the first user (not important)
    if nr_classes % nr_clusters > 0:
        arr_a = np.tile(arr_a, nr_clusters- 1)
    else: 
        arr_a = np.tile(arr_a, nr_clusters)
    
    if nr_classes % nr_clusters > 0: # if there are two different cluster "types", solve again for 2nd
        KK = int(len_b)
        FF = np.floor(len_b/2/nr_clients_per_cluster)
        yy = np.arange(1, nr_clients_per_cluster + 1)
        x = scipy.optimize.fsolve(func, x0=0, args=(yy[:, None], CC, KK, FF))
        li_b = [int(np.floor(FF + np.exp(CC*x*(i**2)))) for i in yy]
        arr_b = np.array(li_b)
        arr_b[0] += KK - np.sum(arr_b)
    else:
        arr_b = np.empty(0)
    
    final_arr = np.concatenate((arr_a, arr_b))

    if plot:
        # Plotting the samples vs clients curve for all clusters
        plt.figure(figsize=(7.5,5))
        plt.plot(final_arr[:240], label='Cluster 1-9')
        plt.plot(final_arr[2160:], linestyle='dashed', label='Cluster 10')
        plt.ylabel('Samples')
        plt.xlabel('Clients')
        #plt.title('No. of samples per client across clusters')
        plt.legend()
        markers = np.array([final_arr[:240].min(), final_arr[2160:].min(), final_arr[:240].max(), final_arr[2161:].max()])
        plt.scatter([0,0,240,240], markers, marker='x')
        plt.annotate(markers[0], (0,markers[0]), textcoords="offset points", xytext=(-2,5), ha='right', fontsize=9)
        plt.annotate(markers[1], (0,markers[1]), textcoords="offset points", xytext=(-2,-5), ha='right', fontsize=9)
        plt.annotate(markers[2], (240,markers[2]), textcoords="offset points", xytext=(-5,-5), ha='right', fontsize=9)
        plt.annotate(markers[3], (240,markers[3]), textcoords="offset points", xytext=(-5,-1), ha='right', fontsize=9)

        plt.show()
        plt.savefig('./all.png')
        
    return final_arr

final_arr = sampling_support_EMNIST(nr_clients, nr_clusters, nr_classes, nr_samples, plot=True)        
print(final_arr)