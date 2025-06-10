#!/usr/bin/env python
import os
import torch
import glob
import argparse

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pth")), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True, help="Current job ID (for sequential training)")
    parser.add_argument("--total_jobs", type=int, required=True, help="Total number of sequential jobs")
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--epochs_per_job", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True, help="Path to output log file")

    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.model_path):
        print(f"Final model {args.model_path} exists. Cancelling remaining jobs.")
        os.system(f"scancel --array={args.job_id+1}-{args.total_jobs} $SLURM_ARRAY_JOB_ID")
        exit(0)

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    cmd_args = [
        "--fasta", args.fasta,
        "--num_layers", "6",
        "--embed_dim", "320",
        "--attention_heads", "20",
        "--epochs", str(args.epochs_per_job),
        "--model_path", args.model_path,
        "--checkpoint_path", args.checkpoint_path
    ]

    if latest_checkpoint:
        print(f"Resuming training from {latest_checkpoint}")
        cmd_args.extend(["--resume_from_checkpoint", latest_checkpoint])

    import subprocess

    cmd = ["python", "-u", "esm2_model.py"] + cmd_args  # -u = unbuffered
    with open(args.log_file, "a") as log_file:
       subprocess.run(cmd, stdout=log_file, stderr=log_file)
