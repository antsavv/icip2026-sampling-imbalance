import numpy as np

np.set_printoptions(linewidth=200)

def calc_class_weights(class_counts, weighting_method):
    """
    Calculate and return class weights based on the class counts and weighting method.

    Args:
        class_counts (np.array): The number of points in each class.
        weighting_method (str): The weighting method to use ('inv_freq', 'class_balanced', 'inv_log', 'inv_pow', 'compl_freq').

    Returns:
        np.array: The calculated class weights.
    """

    total_samples = np.sum(class_counts)
    class_frequencies = class_counts / total_samples

    if weighting_method == 'inv_freq':
        class_weights = 1.0 / class_frequencies

    elif weighting_method == 'class_balanced':
        b = 0.9
        divisor = 1000000
        class_weights = (1 - b) / (1 - np.power(b, class_counts / divisor))

    elif weighting_method == 'compl_freq':
        class_weights = 1.0 - class_frequencies

    elif weighting_method == 'inv_log':
        class_weights = 1.0 / np.log10(class_counts)

    elif weighting_method == 'inv_pow':
        T = 1.0
        gamma = 0.1
        class_weights = 1.0 / np.power(class_counts, gamma / T)

    else:
        raise ValueError(f"Unknown weighting method: {weighting_method}")

    
    class_weights = class_weights / np.sum(class_weights)
    
    return class_weights

def print_weights_for_datasets(datasets):
    methods = ['inv_freq', 'class_balanced', 'inv_log', 'inv_pow', 'compl_freq']
    for dataset_name, dataset in datasets.items():
        print(f"Dataset {dataset_name}:")
        for method in methods:
            weights = calc_class_weights(dataset, method)
            print(f"  {method}: {weights}")
        print()


s3dis = np.array([37334028, 32206900, 53133563, 4719832, 4145093, 4127868, 10681455, 7930065, 6318085, 9188737, 949299, 2457821, 21826246])

dales = np.array([171803872, 120576463, 2567061, 744729, 777716, 1482534, 267847, 56648714])

stpls3d = np.array([847895239, 576172997, 703621125, 33220987, 8391386, 19030897])

datasets = {
    "s3dis": s3dis,
    "dales": dales,
    "stpls3d": stpls3d,
}

print_weights_for_datasets(datasets)