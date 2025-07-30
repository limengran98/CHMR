import time
from tqdm import tqdm

import torch

from .misc import AverageMeter

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
reg_criterion = torch.nn.L1Loss(reduction="none")





def finetune_func(args, model, train_loaders, optimizer, scheduler, epoch):
    if args.task_type == "regression":
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    model.train()
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            batched_data = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batched_data = next(train_loaders["train_iter"])
        batched_data = batched_data.to(device)
        targets = batched_data.y.to(torch.float32)
        is_labeled = targets == targets
        if batched_data.x.shape[0] == 1 or batched_data.batch[-1] == 0:
            continue
        else:
            preds = model(batched_data)
            loss = criterion(
                preds.view(targets.size()).to(torch.float32)[is_labeled],
                targets[is_labeled],
            ).mean()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.steps,
                    # lr=scheduler.get_last_lr()[0],
                    lr=args.lr,
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
            p_bar.update()
    if not args.no_print:
        p_bar.close()

    return train_loaders
