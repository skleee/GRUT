import logging
import random
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from IPython import embed

from utils.prompt import load_prompt_template, get_info_from_prompt, check_task_prompt
from utils import utils, indexing


class MultiTaskDatasetRec(Dataset):
    def __init__(
        self, args, dataset, mode, model_gen, tokenizer, phase=0, regenerate=True
    ):
        super().__init__()
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

        self.user_id_without_target_item = self.args.user_id_without_target_item

        if self.rank == 0:
            logging.info(f"Generating data for {self.dataset} dataset")

        # load and check prompt
        if self.rank == 0:
            logging.info(f"Get prompt template from {args.prompt_file}")
        self.prompt = load_prompt_template(args.prompt_file, self.tasks)

        if self.rank == 0 and "sequential" in self.prompt:  # changed JT
            logging.info(f"{self.prompt['sequential']['seen']['0']['Input']}")
        check_task_prompt(self.prompt, self.tasks)
        self.info = get_info_from_prompt(self.prompt)
        if self.rank == 0:
            logging.info(f"Required info: {self.info}")

        if "history" in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep

        self.user_sequence = utils.ReadLineFromFile(
            os.path.join(self.data_path, self.dataset, "user_sequence.txt")
        )
        self.user_sequence_dict = indexing.construct_user_sequence_dict(
            self.user_sequence
        )

        # apply indexing method and avoid generate data multiple times
        if args.distributed:
            if self.rank == 0:
                logging.info("Reindex data with generative indexing method")
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
                dist.barrier()
            else:
                dist.barrier()
            self.reindex_user_seq_dict, self.item_map = (
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=False,
                    phase=self.phase,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
            )
        else:
            logging.info("Reindex data with generative indexing method")
            self.reindex_user_seq_dict, self.item_map = (
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=self.model_gen,
                    tokenizer=self.tokenizer,
                    regenerate=regenerate,
                    phase=self.phase,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
            )

        self.all_items = list(self.item_map.values())
        # get positive samples for each user to sample negative candidates or evaluation
        self.positive = self.get_positive()

        # load data
        if self.mode == "train":
            if self.rank == 0:
                logging.info("loading training data")
            self.data_samples = self.load_train()

        elif self.mode == "validation":
            self.data_samples = self.load_validation()
            if self.rank == 0:
                logging.info("loading validation data")
            self.valid_prompt = args.valid_prompt
            if self.rank == 0:
                logging.info(f"The validation prompt is {self.valid_prompt}")
        else:
            raise NotImplementedError

        if self.args.debug_train_100:
            self.data_samples = self.data_samples[:100]
            if self.rank == 0:
                logging.info(
                    ">>>> Debug mode: only use 100 samples for training (MultiTaskDatasetRec)"
                )

        # get prompt related info, including numbers and index
        self.get_prompt_info()

        self.construct_sentence()

    def get_positive(self):
        """
        Get a dict of set to save the positive interactions for negative candidate sampling
        """
        positive = dict()
        for user in self.reindex_user_seq_dict:
            if self.mode == "train":
                positive[user] = set(self.reindex_user_seq_dict[user][:-2])
            if self.mode == "validation":
                positive[user] = set(self.reindex_user_seq_dict[user][:-1])
            if self.mode == "test":
                positive[user] = set(self.reindex_user_seq_dict[user])
        return positive

    def shuffle(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        for task in self.task_data:
            indices = torch.randperm(len(self.task_data[task]), generator=g).tolist()
            self.task_data[task] = [self.task_data[task][i] for i in indices]

    def get_prompt_info(self):
        """
        Calculate number of prompts and cumulative index for each task
        - task_prompt_num: save the number of prompts for each task
        - task_index: the cumulative index for each task. if task_index[i-1] <= idx < task_index[i], then the idx belongs to task[i]
            - For example, there are 100 data samples in total, there are 3 tasks, the task_prompt_num is [2,1,3], then the task_index is [200, 300, 600].
        """
        if self.rank == 0:
            logging.info(f"Getting prompt information")
        if self.mode == "train":
            if self.args.sample_prompt == 0:
                self.task_prompt_num = [
                    len(self.prompt[task]["seen"]) for task in self.tasks
                ]
            else:
                sample_number = self.args.sample_num.split(",")
                self.task_prompt_num = [
                    int(sample_number[i]) for i in range(len(self.tasks))
                ]
        else:
            if self.args.valid_prompt_sample == 0:
                self.task_prompt_num = [1] * len(self.tasks)
            else:
                sample_number = self.args.valid_sample_num.split(",")
                self.task_prompt_num = [
                    int(sample_number[i]) for i in range(len(self.tasks))
                ]
        self.task_index = [self.task_prompt_num[0] * len(self.data_samples)]
        for i in range(1, len(self.task_prompt_num)):
            self.task_index.append(
                self.task_index[i - 1]
                + self.task_prompt_num[i] * len(self.data_samples)
            )
        self.task_data = dict()
        for i in range(len(self.tasks)):
            if i == 0:
                start = 0
            else:
                start = self.task_index[i - 1]
            end = self.task_index[i]
            task = self.tasks[i]
            self.task_data[task] = [i for i in range(start, end)]

    def load_train(self):
        """
        Load training data samples

        Already adopting data augmentation
        - Original: A B C D -> E
        - Augmented: A -> B, A B -> C, A B C -> D, A B C D -> E
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user][:-2]
            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                one_sample = dict()
                one_sample["dataset"] = self.dataset
                one_sample["user_id"] = user
                if self.prefix > 0:
                    one_sample["target"] = "item_" + items[i]
                else:
                    one_sample["target"] = items[i]
                if "history" in self.info:
                    history = items[:i]
                    if self.max_his > 0:
                        history = history[-self.max_his :]
                    if self.prefix > 0:
                        one_sample["history"] = self.his_sep.join(
                            ["item_" + item_idx for item_idx in history]
                        )
                    else:
                        one_sample["history"] = self.his_sep.join(history)
                data_samples.append(one_sample)
        return data_samples

    def load_validation(self):
        """
        Load validation data samples
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user]
            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user
            if self.prefix > 0:
                one_sample["target"] = "item_" + items[-2]
            else:
                one_sample["target"] = items[-2]
            if "history" in self.info:
                history = items[:-2]
                if self.max_his > 0:
                    history = history[-self.max_his :]
                if self.prefix > 0:
                    one_sample["history"] = self.his_sep.join(
                        ["item_" + item_idx for item_idx in history]
                    )
                else:
                    one_sample["history"] = self.his_sep.join(history)
            data_samples.append(one_sample)
        return data_samples

    def __len__(self):
        return len(self.data["input"])

    def construct_sentence(self):
        if self.mode == "train":
            if self.args.sample_prompt == 0:
                self._construct_sentence_all()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(
                    f"Input: {self.data['input'][-1]} , Output: {self.data['output'][-1]} "
                )
        elif self.mode == "validation":
            if self.args.valid_prompt_sample == 0:
                self._construct_sentence_valid()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(
                    f"Input: {self.data['input'][-1]} , Output: {self.data['output'][-1]} "
                )

    def _construct_sentence_valid(self):
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        setting = self.valid_prompt.split(":")
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                self.data["input"].append(
                    self.prompt[task][setting[0]][setting[1]]["Input"].format(
                        **datapoint
                    )
                )
                self.data["output"].append(
                    self.prompt[task][setting[0]][setting[1]]["Output"].format(
                        **datapoint
                    )
                )

    def _construct_sentence_all(self):
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        for task in self.tasks:
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for pid in self.prompt[task]["seen"]:
                    self.data["input"].append(
                        self.prompt[task]["seen"][pid]["Input"].format(**datapoint)
                    )
                    self.data["output"].append(
                        self.prompt[task]["seen"][pid]["Output"].format(**datapoint)
                    )

    def _construct_sentence_sample(self):
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]["seen"]) - 1)
                    self.data["input"].append(
                        self.prompt[task]["seen"][str(pid)]["Input"].format(**datapoint)
                    )
                    self.data["output"].append(
                        self.prompt[task]["seen"][str(pid)]["Output"].format(
                            **datapoint
                        )
                    )

    def __getitem__(self, idx):

        return {"input": self.data["input"][idx], "output": self.data["output"][idx]}
