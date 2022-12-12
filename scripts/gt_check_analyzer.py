import pickle
import numpy as np

def remove_outlier(values):
    # z_score = numpy.abs(stats.zscore(values))
    # inlier = values[z_score<3]
    median = np.median(values)
    mad = np.median(abs(values - median))
    z_score = 0.6745*(values - median)/mad
    inlier = values[z_score < 3.5]
    return inlier

if __name__ == '__main__':
    data_filepath = '/data/datasets/DARPA_SubT/anymal_2/gt_check_full_dataset_r01.dat'

    with open(data_filepath, "rb") as data_file:
        command = pickle.load(data_file)
        print(command)
        data = pickle.load(data_file)
        sequence = command[-1]
        sequence = sequence.split("/")[0]
        print(sequence)

    data = np.asarray(data)
    error = data[:,0]

    mean_error = np.mean(error)
    std_error = np.std(error)
    print("Mean error: ", mean_error)
    print("Std error: ", std_error)

    error_filtered = remove_outlier(error)
    mean_error_filtered = np.mean(error_filtered)
    std_error_filtered = np.std(error_filtered)
    print("Mean error filtered: ", mean_error_filtered)
    print("Std error filtered: ", std_error_filtered)

