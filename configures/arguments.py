#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arguments module for CHMR.
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="CHMR pre-training / fine-tuning (no YAML)")

    # Runtime / IO
    parser.add_argument("--gpu-id", type=int, default=0, help="Which GPU to use (default: 0)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no-print", action="store_true", default=False, help="Disable progress bar/log prints")
    parser.add_argument("--dataset", type=str, default="pretrain", help="Dataset name")
    parser.add_argument("--model-path", type=str, default="ckpt/pretrain.pt", help="Path to save/load model checkpoint")

    # Model / GNN
    parser.add_argument("--model", type=str, default="gin-virtual",
                        help="Model type: gin, gin-virtual, gcn, gcn-virtual")
    parser.add_argument("--readout", type=str, default="sum", help="Graph readout function")
    parser.add_argument("--norm-layer", type=str, default="batch_norm", help="Normalization layer type")
    parser.add_argument("--drop-ratio", type=float, default=0.5, help="Dropout ratio")
    parser.add_argument("--num-layer", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--emb-dim", type=int, default=300, help="Hidden dimension in GNNs")

    # Context graph
    parser.add_argument("--walk-length", type=int, default=4, help="Context walk length")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for context graph")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=5120, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--initw-name", type=str, default="default", help="Weight initialization method")

    # Visualization / preprocessing
    parser.add_argument("--vis-plot", action="store_true", help="Enable TSNE visualization")
    parser.add_argument("--fill_method", type=str, default="mean", help="Fill method: mean, zero, nearest")

    # Task-specific (keep original defaults)
    parser.add_argument("--depth", type=int, default=6)                 # d
    parser.add_argument("--lambda_1", type=float, default=0.1)
    parser.add_argument("--lambda_2", type=float, default=0.01)
    parser.add_argument("--ec_ce_weight", type=float, default=1.0)      # μ
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--task_dropout", type=float, default=0.8)

    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to append to model name."
    )
    parser.add_argument("--timestamp", type=str, default="", help="Timestamp to locate unique checkpoint.")

    args = parser.parse_args()

    # Extra default parameter preserved from your code
    args.n_steps = 1

    return args


if __name__ == "__main__":
    # Quick smoke test: prints parsed args (respects --no-print if you handle it in your main script)
    a = get_args()
    print(a)
