import logging
import os
import random
import re
import copy

from time import time
from IPython import embed
from torch.utils.data import Dataset

from utils import indexing
from utils.prompt import (
    load_prompt_template,
    get_info_from_prompt,
    check_task_prompt,
)


class TestDatasetGRAM(Dataset):
    def __init__(
        self,
        args,
        dataset,
        task,
        model_gen,
        tokenizer,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
        mode="test",
    ):
        super().__init__()
        self.args = args
        self.data_path = args.data_path
        self.dataset = dataset
        self.task = task
        self.phase = phase
        self.model_gen = model_gen
        self.tokenizer = tokenizer
        self.mode = mode

        self.reverse_history = args.reverse_history
        self.user_id_without_target_item = args.user_id_without_target_item
        self.id_linking = args.id_linking

        self.prompt = load_prompt_template(args.prompt_file, [self.task])
        check_task_prompt(self.prompt, [self.task])
        self.info = get_info_from_prompt(self.prompt)
        if "history_lex_id" in self.info:
            self.max_his = args.max_his
            self.his_sep = args.his_sep

        self.user_seq_dict, self.item2input, self.item2lexid, _ = indexing.gram_indexing(
            data_path=self.data_path,
            dataset=self.dataset,
            model_gen=self.model_gen,
            tokenizer=self.tokenizer,
            regenerate=regenerate,
            phase=self.phase,
            args=self.args,
            user_id_without_target_item=self.user_id_without_target_item,
            id_linking=self.id_linking,
        )
        self.all_items = list(self.item2lexid.values())

        # load data
        if self.mode == "test":
            self.data_samples = self.load_test()
        elif self.mode == "validation":
            self.data_samples = self.load_validation()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.args.debug_test_100 or debug_test_small_set:
            self.data_samples = self.data_samples[:100]
            if self.args.rank == 0:
                logging.info(
                    ">>>> Debug mode: only use 100 samples for test (TestDatasetGRAM)"
                )

        self.construct_sentence()  # construct 'input', 'output' for each data sample

    def load_test(self):
        """
        Load test data samples

        'dataset': 'Beauty',
        'user_id': 'nail lacquer simmer and shimmer red shatter'
        'target': 'dr scholls quick'
        'history': 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'
        """
        st = time()
        data_samples = []
        for user in self.user_seq_dict:

            items = self.user_seq_dict[user]
            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user  # 'A1YJEY40YUW4SE'
            one_sample["target"] = items[-1]  # 'B004ZT0SSG'
            one_sample["target_lex_id"] = self.item2lexid[
                items[-1]
            ]  # 'red shatter crackle nail polish e55'

            history = items[:-1]
            if self.max_his > 0:
                history = history[-self.max_his :]
            one_sample["history"] = self.his_sep.join(
                history
            )  # 'B00KAL5JAU ; B00KHGIK54'
            one_sample["history_input"] = [
                self.item2input[h] for h in history
            ]  # ['nail lacquer simmer and shimmer', ...]

            if self.reverse_history:
                tmp_history = copy.deepcopy(one_sample["history"]).split(self.his_sep)
                one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                one_sample["history_input"] = one_sample["history_input"][::-1]

            history_lex_ids = [self.item2lexid[h] for h in history[::-1]]
            one_sample["history_lex_id"] = self.his_sep.join(
                history_lex_ids
            )  # 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'

            data_samples.append(one_sample)

        if self.args.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load test time: {time()-st:.2f} s")

        return data_samples

    def load_validation(self):
        """
        Load test valid samples

        'dataset': 'Beauty',
        'user_id': 'nail lacquer simmer and shimmer red shatter'
        'target': 'dr scholls quick'
        'history': 'nail lacquer simmer and shimmer ; red shatter crackle nail polish e55 ; ...'
        """
        st = time()

        data_samples = []
        for user in self.user_seq_dict:
            items = self.user_seq_dict[user]
            one_sample = dict()
            one_sample["dataset"] = self.dataset
            one_sample["user_id"] = user  # 'A1YJEY40YUW4SE'
            one_sample["target"] = items[-2]  # 'B004ZT0SSG'
            one_sample["target_lex_id"] = self.item2lexid[
                items[-2]
            ]  # 'red shatter crackle nail polish e55'

            history = items[:-2]
            if self.max_his > 0:
                history = history[-self.max_his :]
            one_sample["history"] = self.his_sep.join(
                history
            )  # 'B00KAL5JAU ; B00KHGIK54'
            one_sample["history_input"] = [
                self.item2input[h] for h in history
            ]  # ['nail lacquer simmer and shimmer', ...]

            if self.reverse_history:
                tmp_history = copy.deepcopy(one_sample["history"]).split(self.his_sep)
                one_sample["history"] = self.his_sep.join(tmp_history[::-1])
                one_sample["history_input"] = one_sample["history_input"][::-1]

            history_lex_ids = [self.item2lexid[h] for h in history[::-1]]
            one_sample["history_lex_id"] = self.his_sep.join(history_lex_ids)

            data_samples.append(one_sample)

        if self.args.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Load validation time: {time()-st:.2f} s")

        return data_samples

    def construct_sentence(self):
        """
        Make data_samples into model input, output pairs
        data_samples: {
            'dataset': 'Beauty',
            'user_id': 'A2CG5Y82ZZNY6W',
            'target': 'B00KHH2VOY',
            'target_lex_id': 'dead sea salt soap body soap dead sea',
            'history': 'B00KAL5JAU ; B00KHGIK54',
            'history_input': ['dead sea salt deep hair conditioner natural curly',
                                'adovia facial serum anti aging'],
            'history_lex_id': 'dead sea salt deep hair conditioner natural curly ; adovia facial serum anti aging',
            }
        """
        st = time()
        self.data = {}
        self.data["input"] = []
        self.data["output"] = []
        self.data["user_id"] = []

        for i in range(len(self.data_samples)):
            datapoint = self.data_samples[i]
            input_sample = []

            sequence_text = datapoint["history_lex_id"]
            user_sentence = f"What would user purchase after {sequence_text} ?"
            input_sample.append(user_sentence)

            history_input = datapoint["history_input"]
            input_sample += history_input

            self.data["input"].append(input_sample)
            self.data["output"].append(datapoint["target_lex_id"])
            self.data["user_id"].append(datapoint["user_id"])

        if self.args.rank == 0 and self.args.verbose_input_output:
            logging.info(f">> Constructing sentence time: {time()-st:.2f} s")
            logging.info(
                f">> Input: {self.data['input'][-1]} \n>> Output: {self.data['output'][-1]}"
            )

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.get_item(idx)

    def get_item(self, idx):
        return {
            "input": self.data["input"][idx],
            "output": self.data["output"][idx],
            "user_id": self.data["user_id"][idx],
        }
