import yaml
import argparse


# === Model hyperparameter keys for selective saving/loading ===
model_hyperparams = [
    "emb_dim",
    "model",
    "num_layer",
    "readout",
    "norm_layer",
    "threshold",
    "walk_length",
    "prior",
]


def load_arguments_from_yaml(filename, model_only=False):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    if model_only:
        config = {k: v for k, v in config.items() if k in model_hyperparams}
    return config


def save_arguments_to_yaml(args, filename, model_only=False):
    if model_only:
        args = {k: v for k, v in vars(args).items() if k in model_hyperparams}
    else:
        args = vars(args)

    with open(filename, "w") as f:
        yaml.dump(args, f)


def get_args():
    parser = argparse.ArgumentParser(description="Bayes‑OT‑Tree pre‑training")

    parser.add_argument("--gpu-id", type=int, default=0, help="Which GPU to use (default: 0)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no-print", action="store_true", default=False, help="Disable progress bar")

    parser.add_argument("--dataset", type=str, default="pretrain", help="Dataset name")

    parser.add_argument("--model", type=str, default="gin-virtual",
                        help="Model type: gin, gin-virtual, gcn, gcn-virtual")
    parser.add_argument("--readout", type=str, default="sum", help="Graph readout function")
    parser.add_argument("--norm-layer", type=str, default="batch_norm", help="Normalization layer type")
    parser.add_argument("--drop-ratio", type=float, default=0.5, help="Dropout ratio")
    parser.add_argument("--num-layer", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--emb-dim", type=int, default=300, help="Hidden dimension in GNNs")

    parser.add_argument("--walk-length", type=int, default=4, help="Context walk length")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for context graph")

    parser.add_argument("--batch-size", type=int, default=5120, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")

    parser.add_argument("--initw-name", type=str, default="default", help="Weight initialization method")
    parser.add_argument("--model-path", type=str, default="ckpt/pretrain.pt", help="Path to save/load model")

    parser.add_argument("--vis_plot", type=bool, default=False, help="Enable TSNE visualization")

    parser.add_argument("--fill_method", type=str, default="mean", help="Fill method: mean, zero, nearest")

    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--intra_weight", type=float, default=0.1)
    parser.add_argument("--inter_weight", type=float, default=0.01)
    parser.add_argument("--ec_ce_weight", type=float, default=1.0)

    args = parser.parse_args()

    print("no print", args.no_print)

    # Extra default parameter
    args.n_steps = 1

    return args
