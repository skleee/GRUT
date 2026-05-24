import torch
import os
import logging
import datetime
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp

from IPython import embed
from transformers import T5Config, AutoTokenizer, AutoModelForSeq2SeqLM
from types import MethodType
from undecorated import undecorated

from runner import get_runner
from arguments import create_parser
from model import create_model
from utils import (
    set_seed,
    setup_logging,
    setup_model_path,
    save_args,
    load_model,
    get_dataset_gram,
    get_loader_gram,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(action="ignore")


def distributed_launch(args):
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    mp.spawn(distributed_main, args=(args,), nprocs=ngpus_per_node, join=True)


def distributed_main(local_rank, args):
    args.rank = local_rank
    set_seed(args.seed)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=18000),
        world_size=args.world_size,
        rank=local_rank,
    )
    setup_logging(args)
    setup_model_path(args)

    if args.rank == 0:
        save_args(args)
        logging.info(vars(args))

    device = f"cuda:{local_rank}"
    args.gpu = local_rank

    if "t5" in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        if local_rank == 0:
            logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError

    config.max_seq_len = args.item_prompt_max_len
    config.max_item_num = args.max_his
    config.use_position_embedding = args.use_position_embedding
    config.sample_num = args.sample_num

    model_backbone = AutoModelForSeq2SeqLM.from_pretrained(args.backbone, config=config)
    model_rec = create_model("gram", config=config)
    model_rec.load_t5(model_backbone.state_dict())  #

    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)

    model_gen = AutoModelForSeq2SeqLM.from_pretrained(
        "nandakishormpai/t5-small-machine-articles-tag-generation"
    )
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    args.tokenizer = tokenizer

    if args.rec_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.rec_model_path}")
        model_rec = load_model(model_rec, args.rec_model_path, args, loc=device)
        model_rec.to(device)

    if args.id_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.id_model_path}")
        model_gen = load_model(model_gen, args.id_model_path, args, loc=device)
        model_gen.to(device)

    TrainSetID, TrainSetRec, ValidSet = get_dataset_gram(args, model_gen, tokenizer)
    train_loader_id, train_loader_rec, valid_loader = get_loader_gram(
        args, tokenizer, TrainSetID, TrainSetRec, ValidSet, local_rank
    )

    runner = get_runner(
        "distributed",
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
        local_rank,
    )

    if args.train:
        if local_rank == 0:
            logging.info("Start training")
        runner.train_generator()

        if local_rank == 0:
            logging.info(f"Train done ... Load model from {runner.cur_model_path}")
        runner.test(runner.cur_model_path)
    dist.barrier()
    dist.destroy_process_group()

    return


def single_main(args):
    setup_logging(args)
    setup_model_path(args)
    set_seed(args.seed)

    args.rank = 0
    device = torch.device("cuda", int(args.gpu.split(",")[0]))

    save_args(args)
    logging.info(vars(args))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    if "t5" in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError

    config.max_seq_len = args.item_prompt_max_len
    config.max_item_num = args.max_his
    config.use_position_embedding = args.use_position_embedding
    config.sample_num = args.sample_num

    model_backbone = AutoModelForSeq2SeqLM.from_pretrained(args.backbone, config=config)
    model_rec = create_model("gram", config=config)
    model_rec.load_t5(model_backbone.state_dict())

    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)

    # TODO: Not used, to be removed (model_gen epoch=0) ----------
    model_gen = AutoModelForSeq2SeqLM.from_pretrained(
        "nandakishormpai/t5-small-machine-articles-tag-generation"
    )
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    args.tokenizer = tokenizer

    TrainSetID, TrainSetRec, ValidSet = get_dataset_gram(
        args, model_gen, tokenizer, phase=0, regenerate=False
    )
    train_loader_id, train_loader_rec, valid_loader = get_loader_gram(
        args, tokenizer, TrainSetID, TrainSetRec, ValidSet
    )

    if args.rec_model_path:
        model_rec = load_model(model_rec, args.rec_model_path, args, loc=device)
        logging.info(f"Load recommender model from {args.rec_model_path}")

    if args.id_model_path:
        model_gen = load_model(model_gen, args.id_model_path, args, loc=device)
        logging.info(f"Load generator model from {args.id_model_path}")

    runner = get_runner(
        "single",
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
    )

    if args.train:
        logging.info("Start training")
        runner.train_generator()

        logging.info(f"Train done ... Load model from {runner.cur_model_path}")
        runner.args.debug_test_100 = 0
        runner.test(runner.cur_model_path)
    else:
        if args.test_by_valid:
            logging.info(
                f"[VALID] Test model from {args.rec_model_path}, {args.id_model_path}"
            )
            runner.validate_from_model(args.rec_model_path, args.id_model_path)
        else:
            logging.info(f"Test model from {args.rec_model_path}, {args.id_model_path}")
            runner.test_from_model(args.rec_model_path, args.id_model_path)


if __name__ == "__main__":
    parser = create_parser()
    init_args, _ = parser.parse_known_args()
    ngpus_per_node = torch.cuda.device_count()
    if init_args.distributed and ngpus_per_node > 1:
        distributed_launch(args=init_args)
    else:
        single_main(args=init_args)
