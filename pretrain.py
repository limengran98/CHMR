import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from configures.arguments import get_args
from dataset.create_datasets import get_data
from dataset.data_utils import align_and_fill_modalities, align_and_aug_modalities
from utils import init_weights
from models.pretrain_model import pretrain_func
from utils.training_utils import get_logger, get_cosine_schedule_with_warmup


from models.pretrain_model import GNN
from torch.distributions import Normal, Independent

def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    dataset, context_graph = get_data(args, "./raw_data", transform="pyg")
    aligned_data = align_and_fill_modalities("./raw_data/pretrain/raw", fill_method=args.fill_method)
    aug_data = align_and_aug_modalities("./raw_data/pretrain/raw")
    context_graph = context_graph[0]


    split_idx = dataset.get_idx_split()
    args.num_trained = len(split_idx["train"])
    args.task_type = dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = GNN(
        gnn_type=args.model,
        # num_tasks=dataset.num_tasks,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.readout,
        norm_layer=args.norm_layer,
        depth=args.depth,
        ec_ce_weight=args.ec_ce_weight,
    ).to(device)
    model.load_pretrained_graph_encoder("./ckpt/pretrain2D.pt")

    init_weights(model, args.initw_name, init_gain=0.02)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    prior_mu = torch.zeros(args.emb_dim).to(device)
    prior_sigma = torch.ones(args.emb_dim).to(device)
    args.prior_dist = Independent(Normal(loc=prior_mu, scale=prior_sigma), 1)
 

    # scheduler = None
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)
    
    logging.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info(model)
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(split_idx['valid'])}/{len(split_idx['test'])}"
    )
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")

    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    best_train, best_valid, best_test, best_count = None, None, None, None
    best_epoch = 0
    loss_tots = []

    for epoch in range(0, args.epochs):
        loss, train_loaders = pretrain_func(
            args, model, train_loaders, aligned_data, aug_data, context_graph, optimizer, scheduler, epoch
        )
        loss_tots.append(loss)
        # === Periodic checkpointing ===
        is_last_epoch = (epoch == args.epochs - 1)
        is_checkpoint_epoch = ((epoch + 1) % 50 == 0)

        if is_checkpoint_epoch and not is_last_epoch:
            ckpt_epoch_path = args.model_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_epoch_path)
            logger.info(f"💾 Checkpoint saved at {ckpt_epoch_path}")

        # === Final save (only once at last epoch) ===
        if is_last_epoch:
            torch.save(model.state_dict(), args.model_path)
            avg_loss = sum(loss_tots) / len(loss_tots)
            logger.info(f"✅ Finished training. Model saved at {args.model_path}. Final avg loss = {avg_loss:.4f}")

    return (
        args.model_path,
        args.dataset,
        dataset.eval_metric,
        best_train,
        best_valid,
        best_test,
        best_epoch,
        best_count,
    )


if __name__ == "__main__":
    import os
    import datetime

    args = get_args()

    # === Auto-generate model checkpoint name based on hyperparameters ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_tag = (
        f"CHMR"
        f"-lr={args.lr}"
        f"-wdecay={args.wdecay}"
        f"-epoch={args.epochs}"
        f"-batch={args.batch_size}"
        f"-lambda1={args.lambda_1}"
        f"-lambda2={args.lambda_2}"
        f"-decomp={args.decomp_method}"
    )

    # Append user note if provided
    if hasattr(args, "note") and args.note:
        model_tag += f"-{args.note}"

    # Append timestamp for uniqueness
    model_tag += f"-{timestamp}"

    ckpt_dir = "ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    args.model_path = os.path.join(ckpt_dir, model_tag + ".pt")

    # === Auto-detect previous checkpoint ===
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("CHMR") and f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
        reverse=True,
    )
    args.resume_ckpt = None

    if ckpt_files:
        latest_ckpt = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"🔍 Detected existing checkpoint: {latest_ckpt}")
        user_choice = input("Do you want to resume training from this checkpoint? (y/n): ").strip().lower()
        if user_choice == "y":
            args.resume_ckpt = latest_ckpt
            print(f"✅ Resuming from checkpoint: {args.resume_ckpt}")
        else:
            print("🚀 Starting a new training run from scratch.")
    else:
        print("🆕 No previous checkpoint found. Starting fresh training.")
        args.resume_ckpt = None

    # === Initialize logger ===
    logger = get_logger(__name__)
    args.logger = logger

    print(f"💾 Model will be saved to: {args.model_path}")
    print(f"📘 Resume checkpoint: {args.resume_ckpt}")
    print(vars(args))

    # === Run main training ===
    main(args, 0)

