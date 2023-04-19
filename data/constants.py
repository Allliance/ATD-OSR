OSR_DATASETS = ['cifar10', 'cifar100', 'mnist', 'fmnist', 'svhn']

def get_run_name(in_dataset, run_index):
    return f'{in_dataset}_{args.run_index}'