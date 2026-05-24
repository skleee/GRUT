import random
import os
import torch
from torch.utils.data import Dataset
from utils.prompt import load_prompt_template, check_task_prompt, get_info_from_prompt
from utils.utils import ReadLineFromFile
from utils import indexing

import torch.distributed as dist
import logging


class MultiTaskDatasetGen(Dataset):
    def __init__(self, args, dataset, mode, phase=0):
        super().__init__()
        self.data_path = args.data_path
        self.dataset = dataset
        self.tasks = args.tasks.split(",")
        if args.sample_prompt > 0:
            assert len(self.tasks) == len(
                args.sample_num.split(",")
            ), "prompt sample number does not match task number"
        self.mode = mode
        self.args = args

        self.rank = args.rank
        self.skip_empty_his = args.skip_empty_his

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
            self.max_his = int(args.max_his / 2)
            self.his_sep = args.his_sep

        # load user sequence data
        self.user_sequence = ReadLineFromFile(
            os.path.join(self.data_path, self.dataset, "user_sequence.txt")
        )
        self.user_sequence_dict = indexing.construct_user_sequence_dict(
            self.user_sequence
        )

        self.user_id_without_target_item = self.args.user_id_without_target_item

        # apply indexing method and avoid generate data multiple times
        if args.distributed:
            self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_id(
                data_path=self.data_path,
                dataset=self.dataset,
                user_sequence_dict=self.user_sequence_dict,
                phase=phase,
                user_id_without_target_item=self.user_id_without_target_item,
                args=args,
            )
            if self.rank == 0:
                logging.info("Reindex data with generative indexing method")
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=None,
                    tokenizer=None,
                    regenerate=False,
                    phase=phase,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
                logging.info(
                    "Successfully reindex data with generative indexing method"
                )
                dist.barrier()
            else:
                dist.barrier()
            if self.rank == 0:
                logging.info("Reindex rec data with generative indexing method")
            self.reindex_user_seq_dict_rec, self.item_map_rec = (
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=None,
                    tokenizer=None,
                    regenerate=False,
                    phase=phase,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
            )
            if self.rank == 0:
                logging.info(
                    "Successfully reindex rec data with generative indexing method"
                )
        else:
            logging.info("Reindex data with generative indexing method")
            # reindex_user_seq_dict: {uid: [item_text, item_text, item_text...]}, item_map: {original iid: item_text}
            # reindex_user_seq_dict_rec: {uid: [iid, iid, iid...]}, item_map_rec: {original iid: iid}
            self.reindex_user_seq_dict, self.item_map = indexing.generative_indexing_id(
                data_path=self.data_path,
                dataset=self.dataset,
                user_sequence_dict=self.user_sequence_dict,
                phase=phase,
                user_id_without_target_item=self.user_id_without_target_item,
                args=args,
            )
            self.reindex_user_seq_dict_rec, self.item_map_rec, self.user_map_rec = (
                indexing.generative_indexing_rec(
                    data_path=self.data_path,
                    dataset=self.dataset,
                    user_sequence_dict=self.user_sequence_dict,
                    model_gen=None,
                    tokenizer=None,
                    regenerate=False,
                    phase=phase,
                    return_user_map=True,
                    user_id_without_target_item=self.user_id_without_target_item,
                    args=args,
                )
            )
        self.all_items = list(self.item_map.values())

        # load data
        if self.mode == "train":
            if self.rank == 0:
                logging.info("loading training data for id generator")
            self.data_samples = self.load_train_id()
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
                    ">>>> Debug mode: only use 100 samples for training (MultiTaskDatasetGen)"
                )

        self.get_prompt_info()
        self.generate_data()

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

    def load_train_id(self):
        """
        Load training data samples

        Adopting data augmentation
        - Original: A B C D -> E
        - Augmented: A -> B, A B -> C, A B C -> D, A B C D -> E
        """
        data_samples = []
        for user in self.reindex_user_seq_dict:
            items = self.reindex_user_seq_dict[user][
                :-2
            ]  # metadata ([description: opi burlesque colors; categories: beauty, makeup, nails, nail polish; price: 12.0; title: opi nail lacquer, simmer and shimmer, 0.5fluid ounce; brand: opi; salesrank: beauty: 46572', ...])
            items_rec = self.reindex_user_seq_dict_rec[user][
                :-2
            ]  # item id (['nail lacquer simmer and shimmer','red shatter crackle nail polish e55', 'skin79 the prestige beblesh balm']  )

            for i in range(len(items)):
                if i == 0:
                    if self.skip_empty_his > 0:
                        continue
                # add [len_hist, len_maxtoken] plain text of user history
                # add [len_hist] binary mask
                # add prompt
                # self.user_id = user
                one_sample = dict()
                one_sample["dataset"] = self.dataset
                one_sample["user_id"] = user

                one_sample["target"] = items_rec[i]
                if "history" in self.info:
                    history = items[:i]
                    if self.max_his > 0:
                        history = history[-self.max_his :]
                    one_sample["history"] = history

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
        return len(self.data["input_prompt"])

    def construct_sentence(self):
        if self.mode == "train":
            if self.args.sample_prompt == 0:
                self._construct_sentence_all()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(
                    f"Input: {self.data['input'][100]} , Output: {self.data['output'][100]} "
                )
        elif self.mode == "validation":
            if self.args.valid_prompt_sample == 0:
                self._construct_sentence_valid()
            else:
                self._construct_sentence_sample()
            if self.rank == 0:
                logging.info(
                    f"Input: {self.data['input'][100]} , Output: {self.data['output'][100]} "
                )
                logging.info(
                    f"Input: {self.data['input'][101]} , Output: {self.data['output'][101]} "
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

    def generate_data(self):
        """"""
        self.data = {}
        self.data["history"] = []
        self.data["target"] = []
        self.data["input_prompt"] = []
        self.data["output_prompt"] = []

        for t in range(len(self.tasks)):
            task = self.tasks[t]
            for i in range(len(self.data_samples)):
                datapoint = self.data_samples[i]
                for j in range(self.task_prompt_num[t]):
                    pid = random.randint(0, len(self.prompt[task]["seen"]) - 1)
                    dataset = datapoint["dataset"]
                    user_id = datapoint["user_id"]
                    target = datapoint["target"]
                    self.data["history"].append(datapoint["history"])
                    self.data["target"].append(target)

                    input_prompt = self.prompt[task]["seen"][str(pid)]["Input"].format(
                        **{
                            "dataset": dataset,
                            "user_id": user_id,
                            "history": "{history}",
                        }
                    )
                    output_prompt = self.prompt[task]["seen"][str(pid)][
                        "Output"
                    ].format(**{"dataset": dataset, "target": target})

                    self.data["input_prompt"].append(input_prompt)
                    self.data["output_prompt"].append(output_prompt)
        return True

    def __getitem__(self, idx):

        return {
            "history": self.data["history"][idx],
            "target": self.data["target"][idx],
            "input_prompt": self.data["input_prompt"][idx],
            "output_prompt": self.data["output_prompt"][idx],
        }
