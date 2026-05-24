import numpy as np
import os
import logging
import sys
import random
import torch
import json
from time import strftime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path, "r") as fd:
        for line in fd:
            lines.append(line.rstrip("\n"))
    return lines


def setup_logging(args):
    if len(args.datasets.split(",")) > 1:
        folder_name = "SP5"
    else:
        folder_name = args.datasets
    folder = os.path.join(args.log_dir, folder_name)

    # Using os.makedirs to avoid race condition issues
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Add index to each log_dir
    folder_dirs = os.listdir(folder)
    folder_dirs = [
        d
        for d in folder_dirs
        if os.path.isdir(os.path.join(folder, d)) and d.split("_")[0].isdigit()
    ]  # Only keep dirs with index
    if len(folder_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split("_")[0]) for d in folder_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = f"{idx}_{strftime('%Y%m%d_%H%M')}"
    full_log_dir = os.path.join(folder, cur_log_dir)
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir, exist_ok=True)
    args.log_name = log_name(args)

    log_file = os.path.join(full_log_dir, args.log_name + ".log")
    args.full_log_dir = full_log_dir  # save the full log dir to args

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    return


def save_args(args):
    args_vars = vars(args)
    with open(os.path.join(args.full_log_dir, "config.json"), "w") as f:
        json.dump(args_vars, f, indent=4, default=str, ensure_ascii=False)


def save_tokenizer(args, tokenizer):
    if not os.path.exists(os.path.join(args.full_log_dir, "tokenizer")):
        os.makedirs(os.path.join(args.full_log_dir, "tokenizer"), exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.full_log_dir, "tokenizer"))


def log_name(args):
    if len(args.datasets.split(",")) > 1:
        folder_name = "SP5"
    else:
        folder_name = args.datasets
    params = [
        str(args.distributed),
        str(args.sample_prompt),
        str(args.skip_empty_his),
        str(args.max_his),
        str(args.master_port),
        folder_name,
        args.tasks,
        args.backbone.replace("/", "_"),
        str(args.batch_size),
        args.sample_num,
        args.prompt_file[3:-4],
    ]
    return "_".join(params)


def setup_model_path(args):
    args.model_path = os.path.join(
        args.full_log_dir, f"id_{args.id_epochs}_rec_{args.rec_epochs}"
    )

    from pathlib import Path

    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    return


def save_model(model, path):
    torch.save(model.state_dict(), path)
    return


def load_model(model, path, args, loc=None):
    if loc is None and hasattr(args, "gpu"):
        gpuid = args.gpu.split(",")
        loc = f"cuda:{gpuid[0]}"
    state_dict = torch.load(path, map_location="cpu")
    model.to(loc)
    model.load_state_dict(state_dict, strict=False)
    return model
