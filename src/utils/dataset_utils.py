import torch
from IPython import embed

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import MultiTaskDatasetGRAM
from processor import (
    DistMultiDataTaskSampler,
    SingleMultiDataTaskSampler,
    CollatorGen,
    Collator,
    CollatorGRAM,
)


def get_dataset_gram(args, model_gen, tokenizer, phase=0, regenerate=False):
    datasets = args.datasets.split(",")
    train_all_datasets_id = []
    train_all_datasets_rec = []
    valid_all_datasets = []
    for data in datasets:
        TrainDatasetRec = MultiTaskDatasetGRAM(
            args, data, "train", model_gen, tokenizer, phase, regenerate=regenerate
        )
        TrainDatasetID = MultiTaskDatasetGRAM(
            args, data, "train", model_gen, tokenizer, phase, regenerate=regenerate
        )

        train_all_datasets_id.append(TrainDatasetID)
        train_all_datasets_rec.append(TrainDatasetRec)

    TrainSetID = ConcatDataset(
        train_all_datasets_id
    )  # Not used (Just for compatibility)
    TrainSetRec = ConcatDataset(train_all_datasets_rec)
    ValidSet = None

    return TrainSetID, TrainSetRec, ValidSet


def get_loader_gram(
    args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0, model_config=None
):
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = (
            DistMultiDataTaskSampler(
                TrainSetID,
                args.id_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetID, args.id_batch_size, args.seed, shuffle=True
            )
        )
        train_sampler_rec = (
            DistMultiDataTaskSampler(
                TrainSetRec,
                args.rec_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetRec, args.rec_batch_size, args.seed, shuffle=True
            )
        )
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = (
            DistributedSampler(TrainSetRec) if args.distributed else None
        )
    collator_gen = CollatorGen(tokenizer=tokenizer)
    collator_rec = CollatorGRAM(tokenizer=tokenizer, args=args, mode="train")
    train_loader_id = DataLoader(
        dataset=TrainSetID,
        sampler=train_sampler_id,
        batch_size=args.id_batch_size,
        collate_fn=collator_gen,
        shuffle=False,
    )
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=args.rec_batch_size,
        collate_fn=collator_rec,
        shuffle=False,
    )
    valid_loader = None

    return train_loader_id, train_loader_rec, valid_loader


# temporal for args.debug_test_on_train - distributed not supported
def get_loader_gram_train(args, tokenizer, TrainSetRec):
    train_sampler_rec = SingleMultiDataTaskSampler(
        TrainSetRec, 1, args.seed, shuffle=True
    )
    collator_rec = CollatorGRAM(tokenizer=tokenizer, args=args, mode="test")
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=1,
        collate_fn=collator_rec,
        shuffle=False,
    )
    return train_loader_rec


def get_dataset_grut(args, model_gen, tokenizer, phase=0, regenerate=False):
    from data.multi_task_dataset_grut import MultiTaskDatasetGRUT  # lazy to avoid circular
    datasets = args.datasets.split(',')
    train_all_datasets_id = []
    train_all_datasets_rec = []
    for data in datasets:
        TrainDatasetRec = MultiTaskDatasetGRUT(
            args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate,
        )
        TrainDatasetID = MultiTaskDatasetGRUT(
            args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate,
        )
        train_all_datasets_id.append(TrainDatasetID)
        train_all_datasets_rec.append(TrainDatasetRec)

    TrainSetID = ConcatDataset(train_all_datasets_id)
    TrainSetRec = ConcatDataset(train_all_datasets_rec)
    ValidSet = None
    return TrainSetID, TrainSetRec, ValidSet


def get_loader_grut(args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0, model_config=None):
    from processor import CollatorGRUT  # lazy
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = (
            DistMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.world_size, rank, args.seed, shuffle=True)
            if args.distributed
            else SingleMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.seed, shuffle=True)
        )
        train_sampler_rec = (
            DistMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.world_size, rank, args.seed, shuffle=True)
            if args.distributed
            else SingleMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.seed, shuffle=True)
        )
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = DistributedSampler(TrainSetRec) if args.distributed else None

    collator_gen = CollatorGen(tokenizer=tokenizer)
    collator_rec = CollatorGRUT(tokenizer=tokenizer, args=args, mode='train')

    train_loader_id = DataLoader(
        dataset=TrainSetID, sampler=train_sampler_id,
        batch_size=args.id_batch_size, collate_fn=collator_gen, shuffle=False,
    )
    train_loader_rec = DataLoader(
        dataset=TrainSetRec, sampler=train_sampler_rec,
        batch_size=args.rec_batch_size, collate_fn=collator_rec, shuffle=False,
    )
    valid_loader = None

    return train_loader_id, train_loader_rec, valid_loader


def get_loader(args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0):
    # generate training validation loader.
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = (
            DistMultiDataTaskSampler(
                TrainSetID,
                args.id_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetID, args.id_batch_size, args.seed, shuffle=True
            )
        )
        train_sampler_rec = (
            DistMultiDataTaskSampler(
                TrainSetRec,
                args.rec_batch_size,
                args.world_size,
                rank,
                args.seed,
                shuffle=True,
            )
            if args.distributed
            else SingleMultiDataTaskSampler(
                TrainSetRec, args.rec_batch_size, args.seed, shuffle=True
            )
        )
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = (
            DistributedSampler(TrainSetRec) if args.distributed else None
        )

    collator_gen = CollatorGen(tokenizer)
    collator_rec = Collator(tokenizer)
    train_loader_id = DataLoader(
        dataset=TrainSetID,
        sampler=train_sampler_id,
        batch_size=args.id_batch_size,
        collate_fn=collator_gen,
        shuffle=False,
    )
    train_loader_rec = DataLoader(
        dataset=TrainSetRec,
        sampler=train_sampler_rec,
        batch_size=args.rec_batch_size,
        collate_fn=collator_rec,
        shuffle=False,
    )
    valid_loader = None

    return train_loader_id, train_loader_rec, valid_loader
