"""
run_experiments.py

Script to run all NER training experiments to compare:
1. Training data impact: dev set vs full dataset
2. Preprocessing impact: original vs preprocessed data

This will create 4 models total:
- ner_ready + dev set
- ner_ready + full dataset
- ner_ready_preprocessed + dev set
- ner_ready_preprocessed + full dataset
"""

import subprocess
import sys
import time


def run_experiment(data_folder, train_file, model_suffix, **kwargs):
    """Run a single training experiment"""

    cmd = [
        sys.executable,
        "mbert_train_configurable.py",
        "--data_folder",
        data_folder,
        "--train_file",
        train_file,
        "--model_suffix",
        model_suffix,
    ]

    # Add any additional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"\n{'=' * 60}")
    print(f"Starting experiment: {model_suffix}")
    print(f"Data folder: {data_folder}")
    print(f"Training file: {train_file}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        # This will show all output in real-time
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Experiment {model_suffix} completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment {model_suffix} failed with return code {e.returncode}")
        return False

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️  Experiment {model_suffix} took {duration:.2f} seconds\n")

    return True


def main():
    """Run all experiments"""

    experiments = [
        # Test training data impact with original data
        {
            "data_folder": "ner_ready",
            "train_file": "ground_truth_dev_set_labels.json",
            "model_suffix": "original_devset",
        },
        {
            "data_folder": "ner_ready",
            "train_file": "weakly_labelled.json",
            "model_suffix": "original_fullset",
        },
        # Test training data impact with preprocessed data
        {
            "data_folder": "ner_ready_preprocessed",
            "train_file": "ground_truth_dev_set_labels.json",
            "model_suffix": "preprocessed_devset",
        },
        {
            "data_folder": "ner_ready_preprocessed",
            "train_file": "weakly_labelled.json",
            "model_suffix": "preprocessed_fullset",
        },
    ]

    print("Starting NER Training Experiments")
    print(f"Total experiments to run: {len(experiments)}")

    successful_experiments = 0
    failed_experiments = []

    overall_start_time = time.time()

    for i, experiment in enumerate(experiments, 1):
        print(f"\n🚀 Running experiment {i}/{len(experiments)}")

        success = run_experiment(**experiment)

        if success:
            successful_experiments += 1
        else:
            failed_experiments.append(experiment["model_suffix"])

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {len(failed_experiments)}")
    print(
        f"Total time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)"
    )

    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")

    print("\nModel outputs will be in:")
    print("  - address_parser_original_devset")
    print("  - address_parser_original_fullset")
    print("  - address_parser_preprocessed_devset")
    print("  - address_parser_preprocessed_fullset")

    print("\nExperiments complete! 🎉")


if __name__ == "__main__":
    main()
