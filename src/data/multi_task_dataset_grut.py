import copy
import os
import torch
import pandas as pd
import logging
import torch.distributed as dist

from torch.utils.data import Dataset
from datetime import datetime
from collections import Counter
from time import time

from utils import indexing
from utils import utils as _utils
from utils.indexing import get_dict_from_lines
from utils.time import make_datetime_to_str, make_diff_to_str


class MultiTaskDatasetGRUT(Dataset):
    def __init__(
        self, args, dataset, mode, model_gen, tokenizer, phase=0, regenerate=False
    ):
        super().__init__()
        assert (
            regenerate == False
        ), "regenerate is not supported in MultiTaskDatasetGRUT. id is fixed"
        assert args.sample_num

        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(",")
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(
                args.sample_num.split(",")
            ), "prompt sample number does not match task number"
        self.mode = mode
        self.args = args
        self.phase = phase

        self.rank = args.rank
        self.skip_empty_his = args.skip_empty_his
        self.reverse_history = args.reverse_history
        self.user_id_without_target_item = args.user_id_without_target_item
        self.id_linking = args.id_linking

        # GRUT-specific attributes
        self.temporal_feature = args.temporal_feature
        self.global_context = args.global_context
        self.global_context_last_N = args.global_context_last_N
        self.global_context_top_K = args.global_context_top_K
        self.global_context_epsilon = args.global_context_epsilon
        self.global_context_decay_tau = args.global_context_decay_tau
        self.global_context_decay_clip = args.global_context_decay_clip
        self.prompt_sentence_for_fid_input = args.prompt_sentence_for_fid_input

        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")

        self.max_his = args.max_his
        self.his_sep = args.his_sep

        # apply indexing method and avoid generate data multiple times
        if args.distributed:
            if self.rank == 0:
                logging.info("Reindex data with generative indexing method")
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                    compute_transition=True,
                    decay_tau=self.global_context_decay_tau,
                    decay_clip=self.global_context_decay_clip,
                )
                dist.barrier()
            else:
                dist.barrier()
            self.user_seq_dict, self.item2input, self.item2lexid, self.srciid2tgtiid2cnt = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                    compute_transition=True,
                    decay_tau=self.global_context_decay_tau,
                    decay_clip=self.global_context_decay_clip,
                )
            )
        else:
            logging.info("Reindex data with generative indexing method")
            self.user_seq_dict, self.item2input, self.item2lexid, self.srciid2tgtiid2cnt = (
                indexing.gram_indexing(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    args=self.args,
                    user_id_without_target_item=self.user_id_without_target_item,
                    id_linking=self.id_linking,
                    compute_transition=True,
                    decay_tau=self.global_context_decay_tau,
                    decay_clip=self.global_context_decay_clip,
                )
            )
        self.all_items = list(self.item2lexid.values())

        # Load user lexical id mapping
        self.user2lexid = get_dict_from_lines(_utils.ReadLineFromFile(
            os.path.join(self.data_path, self.dataset, "user_generative_index_phase_0.txt")
        ))

        # Load temporal feature files
        df_time = pd.read_csv(
            os.path.join(self.data_path, self.dataset, "temporal_feature.tsv"), sep="\t"
        )
        uidiid = list(zip(df_time["user_id"], df_time["item_id"]))
        self.uidiid2abstime = dict(zip(uidiid, df_time["absolute_time"]))
        self.uidiid2reltime = dict(zip(uidiid, df_time["relative_time"]))

        df_time = pd.read_csv(
            os.path.join(self.data_path, self.dataset, "timestamp.txt"), sep="\t"
        )
        uidiid = list(zip(df_time["user_id"], df_time["item_id"]))
        self.uidiid2datetime = dict(zip(uidiid, df_time["datetime"]))

        # load data
        if self.mode == "train":
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()
        elif self.mode == "validation":
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
        else:
            raise NotImplementedError

        if self.args.debug_train_100:
            self.data_samples = self.data_samples[:100]
            if self.rank == 0:
                logging.info(
                    ">>>> Debug mode: only use 100 samples for training (MultiTaskDatasetGRUT)"
                )

        self.get_prompt_info()
        self.construct_sentence()

    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.user_seq_dict:
            if self.mode == "train":
                positive[user] = set(self.user_seq_dict[user][:-2])
            if self.mode == "validation":
                positive[user] = set(self.user_seq_dict[user][:-1])
            if self.mode == "test":
                positive[user] = set(self.user_seq_dict[user])
        return positive

    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]

    def get_prompt_info(self):
        """
        Simplified version for single task with hardcoded prompts
        """
        # For single task, just create simple task_data mapping
        self.task_data = {self.tasks[0]: list(range(len(self.data_samples)))}
        # These might not be needed for single task, but keeping for compatibility
        self.task_prompt_num = [1]
        self.task_index = [len(self.data_samples)]

    def load_train(self):
        """
        Load training data samples.

        already adopting data augmentation (for loop - range(len(items)))
        - Original: A B C D -> E
        - Augmented: A -> B, A B -> C, A B C -> D, A B C D -> E
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user][:-2]

            # Yelp repeat-item disambiguation
            if self.temporal_feature != "none" and self.dataset == "Yelp":
                tmp_counter = Counter()
                item_cnts = []
                for iid in items:
                    tmp_counter[(user, iid)] += 1
                    cnt = tmp_counter[(user, iid)]
                    new_iid = f"{iid}||{cnt}"
                    item_cnts.append(new_iid)

            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample["dataset"] = self.dataset
                one_sample["user_id"] = user  # 'A1YJEY40YUW4SE'
                one_sample["target"] = items[i]  # 'B004ZT0SSG'
                one_sample["target_lex_id"] = self.item2lexid[
                    items[i]
                ]  # 'red shatter crackle nail polish e55'

                # target absolute time string for the C_u prompt prefix
                if self.temporal_feature != "none":
                    if self.dataset == "Yelp":
                        one_sample["target_abstime"] = make_datetime_to_str(
                            self.uidiid2abstime[(user, item_cnts[i])]
                        )
                    else:
                        one_sample["target_abstime"] = make_datetime_to_str(
                            self.uidiid2abstime[(user, items[i])]
                        )
                else:
                    one_sample["target_abstime"] = None

                history = items[:i]
                if self.max_his > 0:
                    history = history[-self.max_his :]

                if self.temporal_feature != "none" and self.dataset == "Yelp":
                    history_cnts = item_cnts[:i]
                    if self.max_his > 0:
                        history_cnts = history_cnts[-self.max_his :]

                one_sample["history"] = self.his_sep.join(
                    history
                )  # 'B00KAL5JAU ; B00KHGIK54'

                if self.reverse_history:
                    tmp_history = copy.deepcopy(one_sample["history"]).split(
                        self.his_sep
                    )
                    one_sample["history"] = self.his_sep.join(tmp_history[::-1])

                # ---- C_u: user-level temporal history block ----
                if self.prompt_sentence_for_fid_input:
                    one_sample["user_lex_id"] = self.user2lexid[user]

                    if self.temporal_feature == "none":
                        history_lex_ids = [
                            self.item2lexid[h] for h in history[::-1]
                        ]
                    else:
                        history_lex_ids = []
                        if self.dataset == "Yelp":
                            target_time = datetime.fromisoformat(
                                self.uidiid2datetime[(user, item_cnts[i])]
                            )
                        else:
                            target_time = datetime.fromisoformat(
                                self.uidiid2datetime[(user, items[i])]
                            )

                        tmp_history = (
                            history_cnts if self.dataset == "Yelp" else history
                        )
                        for h in tmp_history[::-1]:
                            if self.dataset == "Yelp":
                                lexid = self.item2lexid[h.split("||")[0]]
                            else:
                                lexid = self.item2lexid[h]

                            abstime = make_datetime_to_str(
                                self.uidiid2abstime[(user, h)]
                            )
                            reltime = self.uidiid2reltime[(user, h)]

                            if self.temporal_feature == "rel":
                                if reltime != reltime or len(history) == 1:  # NaN check
                                    lexid_time = f"{lexid}"
                                else:
                                    lexid_time = f"{lexid} ({reltime})"
                            elif self.temporal_feature == "abs":
                                lexid_time = f"{lexid} ({abstime})"
                            elif self.temporal_feature == "target_rel":
                                cur_time = datetime.fromisoformat(
                                    self.uidiid2datetime[(user, h)]
                                )
                                interval_str = make_diff_to_str(
                                    target_time - cur_time, target_aware=True
                                )
                                lexid_time = f"{lexid} ({interval_str})"
                            elif self.temporal_feature == "rel_abs":
                                lexid_time = f"{lexid} ({abstime}, {reltime})"
                            elif self.temporal_feature == "target_rel_abs":
                                cur_time = datetime.fromisoformat(
                                    self.uidiid2datetime[(user, h)]
                                )
                                interval_str = make_diff_to_str(
                                    target_time - cur_time, target_aware=True
                                )
                                lexid_time = f"{lexid} ({abstime}, {interval_str})"
                            else:
                                raise NotImplementedError(
                                    f"Temporal feature {self.temporal_feature} is not implemented"
                                )
                            history_lex_ids.append(lexid_time)

                    one_sample["history_lex_id"] = self.his_sep.join(history_lex_ids)
                else:
                    # Ablation: w/o user-level — bare lexids only
                    one_sample["user_lex_id"] = self.user2lexid[user]
                    one_sample["history_lex_id"] = self.his_sep.join(
                        self.item2lexid[h] for h in history[::-1]
                    )

                # ---- C_v: item-level transition block ----
                one_sample["global_context"] = []
                if self.global_context == "none":
                    pass
                else:  # 'all_weight' — paper-final
                    src_ids = history[-self.global_context_last_N :][::-1]
                    tmp_lines = []
                    for src_id in src_ids:
                        if self.global_context_top_K == 0:
                            tmp_lines.append(f"after {self.item2lexid[src_id]}")
                            continue

                        cand_dicts = self.srciid2tgtiid2cnt.get(src_id)
                        if cand_dicts is None:
                            # src never appeared as a source pair in training
                            tmp_lines.append(
                                f"after {self.item2lexid[src_id]}, users often buy:"
                            )
                            continue

                        # Sort by pre-weighted score (τ/c applied at indexing in Task 5)
                        neighbors = sorted(
                            cand_dicts.items(), key=lambda x: x[1], reverse=True
                        )[: self.global_context_top_K]
                        neighbor_lexids = ", ".join(
                            self.item2lexid[tgt] for tgt, _ in neighbors
                        )
                        tmp_lines.append(
                            f"after {self.item2lexid[src_id]}, users often buy: {neighbor_lexids}"
                        )
                    one_sample["global_context"] = tmp_lines

                data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load training data time: {time()-st} s")

        return data_samples

    def load_validation(self):
        """
        Load validation data samples.
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user]

            # Yelp repeat-item disambiguation (must use full sequence to enumerate counts up to items[-2])
            if self.temporal_feature != "none" and self.dataset == "Yelp":
                tmp_counter = Counter()
                item_cnts = []
                for iid in items:
                    tmp_counter[(user, iid)] += 1
                    cnt = tmp_counter[(user, iid)]
                    new_iid = f"{iid}||{cnt}"
                    item_cnts.append(new_iid)

            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user
            one_sample["target"] = items[-2]
            one_sample["target_lex_id"] = self.item2lexid[items[-2]]

            if self.temporal_feature != "none":
                if self.dataset == "Yelp":
                    one_sample["target_abstime"] = make_datetime_to_str(
                        self.uidiid2abstime[(user, item_cnts[-2])]
                    )
                else:
                    one_sample["target_abstime"] = make_datetime_to_str(
                        self.uidiid2abstime[(user, items[-2])]
                    )
            else:
                one_sample["target_abstime"] = None

            history = items[:-2]
            if self.max_his > 0:
                history = history[-self.max_his :]

            if self.temporal_feature != "none" and self.dataset == "Yelp":
                history_cnts = item_cnts[:-2]
                if self.max_his > 0:
                    history_cnts = history_cnts[-self.max_his :]

            one_sample["history"] = self.his_sep.join(history)

            if self.reverse_history:
                tmp_history = copy.deepcopy(one_sample["history"]).split(self.his_sep)
                one_sample["history"] = self.his_sep.join(tmp_history[::-1])

            # ---- C_u: user-level temporal history block ----
            if self.prompt_sentence_for_fid_input:
                one_sample["user_lex_id"] = self.user2lexid[user]

                if self.temporal_feature == "none":
                    history_lex_ids = [self.item2lexid[h] for h in history[::-1]]
                else:
                    history_lex_ids = []
                    if self.dataset == "Yelp":
                        target_time = datetime.fromisoformat(
                            self.uidiid2datetime[(user, item_cnts[-2])]
                        )
                    else:
                        target_time = datetime.fromisoformat(
                            self.uidiid2datetime[(user, items[-2])]
                        )

                    tmp_history = (
                        history_cnts if self.dataset == "Yelp" else history
                    )
                    for h in tmp_history[::-1]:
                        if self.dataset == "Yelp":
                            lexid = self.item2lexid[h.split("||")[0]]
                        else:
                            lexid = self.item2lexid[h]

                        abstime = make_datetime_to_str(self.uidiid2abstime[(user, h)])
                        reltime = self.uidiid2reltime[(user, h)]

                        if self.temporal_feature == "rel":
                            if reltime != reltime or len(history) == 1:  # NaN check
                                lexid_time = f"{lexid}"
                            else:
                                lexid_time = f"{lexid} ({reltime})"
                        elif self.temporal_feature == "abs":
                            lexid_time = f"{lexid} ({abstime})"
                        elif self.temporal_feature == "target_rel":
                            cur_time = datetime.fromisoformat(
                                self.uidiid2datetime[(user, h)]
                            )
                            interval_str = make_diff_to_str(
                                target_time - cur_time, target_aware=True
                            )
                            lexid_time = f"{lexid} ({interval_str})"
                        elif self.temporal_feature == "rel_abs":
                            lexid_time = f"{lexid} ({abstime}, {reltime})"
                        elif self.temporal_feature == "target_rel_abs":
                            cur_time = datetime.fromisoformat(
                                self.uidiid2datetime[(user, h)]
                            )
                            interval_str = make_diff_to_str(
                                target_time - cur_time, target_aware=True
                            )
                            lexid_time = f"{lexid} ({abstime}, {interval_str})"
                        else:
                            raise NotImplementedError(
                                f"Temporal feature {self.temporal_feature} is not implemented"
                            )
                        history_lex_ids.append(lexid_time)

                one_sample["history_lex_id"] = self.his_sep.join(history_lex_ids)
            else:
                one_sample["user_lex_id"] = self.user2lexid[user]
                one_sample["history_lex_id"] = self.his_sep.join(
                    self.item2lexid[h] for h in history[::-1]
                )

            # ---- C_v: item-level transition block ----
            one_sample["global_context"] = []
            if self.global_context == "none":
                pass
            else:  # 'all_weight'
                src_ids = history[-self.global_context_last_N :][::-1]
                tmp_lines = []
                for src_id in src_ids:
                    if self.global_context_top_K == 0:
                        tmp_lines.append(f"after {self.item2lexid[src_id]}")
                        continue

                    cand_dicts = self.srciid2tgtiid2cnt.get(src_id)
                    if cand_dicts is None:
                        tmp_lines.append(
                            f"after {self.item2lexid[src_id]}, users often buy:"
                        )
                        continue

                    neighbors = sorted(
                        cand_dicts.items(), key=lambda x: x[1], reverse=True
                    )[: self.global_context_top_K]
                    neighbor_lexids = ", ".join(
                        self.item2lexid[tgt] for tgt, _ in neighbors
                    )
                    tmp_lines.append(
                        f"after {self.item2lexid[src_id]}, users often buy: {neighbor_lexids}"
                    )
                one_sample["global_context"] = tmp_lines

            data_samples.append(one_sample)

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load validation data time: {time()-st} s")

        return data_samples

    def construct_sentence(self):
        """
        Build the 1+L FiD slot layout for each datapoint.
        Slot 0: C_u prompt (optional date prefix + "What would user purchase after H ?")
        Slots 1..L: C_v lines (one per source item in last-N window)
        """
        st = time()
        self.data = {"input": [], "output": [], "user_id": []}

        for datapoint in self.data_samples:
            input_sample = []
            if self.prompt_sentence_for_fid_input:
                sequence_text = datapoint["history_lex_id"]
                user_sentence = (
                    f"What would user purchase after {sequence_text} ?"
                )
                if (
                    self.temporal_feature == "target_rel_abs"
                    and datapoint.get("target_abstime")
                ):
                    user_sentence = (
                        f"The current date is {datapoint['target_abstime']}. {user_sentence}"
                    )
                input_sample.append(user_sentence)
            # else: prompt_sentence_for_fid_input==0 ablation — skip slot 0

            if self.global_context != "none":
                input_sample += datapoint["global_context"]

            self.data["input"].append(input_sample)
            self.data["output"].append(datapoint["target_lex_id"])
            self.data["user_id"].append(datapoint["user_id"])

        if self.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Constructing sentence time: {time()-st} s")
            logging.info(
                f">> Input: {self.data['input'][-1]} \n>> Output: {self.data['output'][-1]}"
            )

    def __len__(self):
        return len(self.data["input"])

    def __getitem__(self, idx):
        return {
            "input": self.data["input"][idx],
            "output": self.data["output"][idx],
            "user_id": self.data["user_id"][idx],
        }
