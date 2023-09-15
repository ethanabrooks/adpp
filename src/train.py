import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich import print
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import data
import wandb
from evaluate import evaluate, plot
from models import GPT
from optimizer import configure, decay_lr
from pretty import print_row, render_graph


def set_seed(seed: int):
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU), you also need to set the seed for the CUDA device
    # This ensures reproducibility for GPU calculations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python's random module
    random.seed(seed)


def train(
    data_args: dict,
    data_path: Path,
    evaluate_args: dict,
    grad_norm_clip: float,
    log_freq: int,
    lr: float,
    metrics_args: dict,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    optimizer_config: dict,
    run: Optional[Run],
    run_name: str,
    save_freq: int,
    seed: int,
    test_freq: int,
    weights_args: dict,
) -> None:
    save_dir = os.path.join("results", run_name)
    set_seed(seed)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dataset = data.make(data_path, **data_args)

    print("Create net... ", end="", flush=True)
    net = GPT(n_tokens=dataset.n_tokens, step_dim=dataset.step_dim, **model_args).cuda()
    print("✓")

    optimizer = configure(lr=lr, module=net, **optimizer_config)

    counter = Counter()
    n_tokens = 0
    tick = time.time()

    for e in range(n_epochs):
        # Split the dataset into train and test sets
        loader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        print("Loading train data... ", end="", flush=True)
        for t, (sequence, mask) in enumerate(loader):
            step = e * len(loader) + t

            # test
            if t % test_freq == 0:
                df = evaluate(dataset=dataset, net=net, **evaluate_args)

                min_return, max_return = dataset.return_range
                returns = df.groupby("t").mean().returns
                graph = render_graph(*returns, max_num=max_return)
                print("\n" + "\n".join(graph), end="\n\n")
                fig = plot(
                    ymin=min_return,
                    ymax=max_return,
                    df=df,
                )
                *_, final_return = returns
                if run is not None:
                    wandb.log(
                        {
                            "eval/rewards": wandb.Image(fig),
                            "eval/final return": final_return,
                        },
                        step=step,
                    )

            # gradient update
            net.train()
            optimizer.zero_grad()
            weights = dataset.weights(sequence.shape, **weights_args)
            logits, loss = net.forward(sequence, mask, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_clip)
            optimizer.step()

            # update learning rate
            n_tokens += mask.sum()
            final_tokens = (
                n_epochs * len(loader) * n_batch * dataset.step_dim
            )  # number of tokens seen during training
            decayed_lr = decay_lr(lr, final_tokens=final_tokens, n_tokens=n_tokens)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            # log
            log = dataset.get_metrics(
                logits=logits, mask=mask, sequence=sequence, **metrics_args
            )
            counter.update(dict(**log, loss=loss.item()))
            if t % log_freq == 0:
                log = {k: v / log_freq for k, v in counter.items()}
                log.update(lr=decayed_lr, time=(time.time() - tick) / log_freq)
                counter = Counter()
                tick = time.time()
                print_row(dict(step=step, **log), show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log({f"train/{k}": v for k, v in log.items()}, step=step)

            # save
            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
