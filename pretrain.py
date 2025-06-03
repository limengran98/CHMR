import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from configures.arguments import (
    save_arguments_to_yaml,
    get_args,
)
from dataset.create_datasets import get_data
from utils import init_weights

from utils.training_utils import get_logger, get_cosine_schedule_with_warmup

from pretrain_model_V3 import pretrain_func
from pretrain_model_V3 import GNN
from torch.distributions import Normal, Independent

def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    dataset, context_graph = get_data(args, "./raw_data", transform="pyg")
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
        beta=args.beta,
        gamma=args.gamma,
        lambd=args.lambd,
        ec_ce_weight=args.ec_ce_weight,
    ).to(device)
    model.load_pretrained_graph_encoder("/home/lmr/InfoAlign-main/ckpt/pretrain2DIM.pt")
    model.freeze_graph_encoder()

    init_weights(model, args.initw_name, init_gain=0.02)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    prior_mu = torch.zeros(args.emb_dim).to(device)
    prior_sigma = torch.ones(args.emb_dim).to(device)
    args.prior_dist = Independent(Normal(loc=prior_mu, scale=prior_sigma), 1)
 

    # scheduler = None
    total_steps  = args.epochs * args.steps
    args.warmup_steps = int(0.03 * total_steps)


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
            args, model, train_loaders, context_graph, optimizer, scheduler, epoch
        )
        loss_tots.append(loss)
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), args.model_path)
            yaml_path = args.model_path.replace(".pt", ".yaml")
            save_arguments_to_yaml(args, yaml_path, model_only=True)
            logger.info(
                f"Finished Training \n Model saved at {args.model_path} and Arguments saved at {yaml_path} with loss {loss_tots}"
            )

    return (
        args.pretrain_name,
        args.dataset,
        dataset.eval_metric,
        best_train,
        best_valid,
        best_test,
        best_epoch,
        best_count,
    )


if __name__ == "__main__":
    args = get_args()

    pretrain_name = args.model_path.split("/")[-1]
    pretrain_name = pretrain_name.split(".")[0]
    args.pretrain_name = pretrain_name

    # logger = get_logger(__name__, logfile=log_path)
    logger = get_logger(__name__)
    args.logger = logger
    print(vars(args))

    main(args, 0)
