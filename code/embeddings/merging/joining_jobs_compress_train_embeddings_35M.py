# join_all_jobs.py
import numpy as np
import argparse
import os

def main(job_dir, output_file):
    job_arrays = []
    for i in range(10):
        path = os.path.join(job_dir, f"job{i}_embedding_all_35M_our_model.npy")
        print(f"Loading {path}...")
        arr = np.load(path)
        job_arrays.append(arr)

    all_embeddings = np.vstack(job_arrays)
    np.save(output_file, all_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    main(args.job_dir, args.output)

