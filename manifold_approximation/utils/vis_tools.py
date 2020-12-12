import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
def create_acc_loss_graph(model_name, dataset_name, log_root_dir,  results_root_dir='.'):
    contents = open(f'{log_root_dir}/{model_name}.log', 'r').read().split('\n')

    times = []
    train_losses = []
    test_losses = []
    epochs = []

    for c in contents:
        if model_name in c:
            name, timestamp, _, train_loss, _, test_loss, epoch = c.split(',')

            times.append(float(timestamp))
            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
            epochs.append(float(epoch))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax1.plot(epochs,train_losses, label='train_loss')
    ax1.plot(epochs,test_losses, label='test_loss')
    ax1.legend(loc=3) # loc=3 means lower left 
    plt.savefig(f'{results_root_dir}/train_test_graph_{dataset_name}_{model_name}.jpg')