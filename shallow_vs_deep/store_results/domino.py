import numpy as np
import os
import csv

def write_results(seeds,model_tag,dataset_tag,data_correlation,all_train_accuracies, all_val_accuracies,  all_test_acc_m0_c100, all_test_acc_m100_c0,each_seed_file,mean_sd_file):
    print(data_correlation, type(data_correlation))

    # File paths
    mnist=data_correlation.split("c")[0]
    data_correlation=int(mnist[1:])

    # Get the directories
    each_seed_dir = os.path.dirname(each_seed_file)
    mean_sd_dir = os.path.dirname(mean_sd_file)

    # Create directories if they don't exist
    os.makedirs(each_seed_dir, exist_ok=True)
    os.makedirs(mean_sd_dir, exist_ok=True)


    # --- Save individual seed results ---
    each_seed_header = ["dataset", "seed", "Train", "Val", "Test_m0_c100", "Test_m100_c0"]
    each_seed_rows = [
        [data_correlation, seed, f"{train:.2f}", f"{val:.2f}", f"{test1:.2f}", f"{test2:.2f}"]
        for seed, train, val, test1, test2 in zip(
            seeds, all_train_accuracies, all_val_accuracies, all_test_acc_m0_c100, all_test_acc_m100_c0
        )
    ]

    write_header = not os.path.exists(each_seed_file)
    with open(each_seed_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(each_seed_header)
        writer.writerows(each_seed_rows)

    # --- Save mean ± SD results ---
    def format_mean_sd(values):
        mean = np.mean(values)
        std = np.std(values)
        return f"{mean:.2f} ± {std:.2f}"

    mean_row = [
        data_correlation,
        format_mean_sd(all_train_accuracies),
        format_mean_sd(all_val_accuracies),
        format_mean_sd(all_test_acc_m0_c100),
        format_mean_sd(all_test_acc_m100_c0)
    ]

    mean_sd_header = ["dataset", "train", "val", "test_m0_c100", "test_m100_c0"]
    write_header = not os.path.exists(mean_sd_file)

    with open(mean_sd_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(mean_sd_header)
        writer.writerow(mean_row)
