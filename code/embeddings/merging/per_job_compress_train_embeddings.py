# join_job_embeddings.py
import argparse
import os
import numpy as np
import glob
from multiprocessing import Pool, cpu_count

def load_embeddings(npz_file):
    data = np.load(npz_file)
    return [data[k] for k in data.files]  # list of (320,) arrays

def process_file(path):
    try:
        return load_embeddings(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def main(job_id, input_dir, output_path, num_workers):
    pattern = os.path.join(input_dir, f"job{job_id}_gpu*_batch*.npz")
    files = sorted(glob.glob(pattern))

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, files)

    # Flatten the list of lists into one array
    all_embeddings = np.vstack([emb for sublist in results for emb in sublist])
    np.save(output_path, all_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    main(args.job_id, args.input_dir, args.output, args.workers)

