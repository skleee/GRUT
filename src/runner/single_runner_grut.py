import os
import logging
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from time import time

import utils.generation_trie as gt
import utils.evaluate as evaluate

from data.test_dataset_grut import TestDatasetGRUT
from utils import get_loader_gram_train
from utils.trend import log_max_trend_score
from processor import CollatorGRUT


class SingleRunnerGRUT:
    def __init__(
        self,
        model_rec,
        model_gen,
        tokenizer,
        train_loader_id,
        train_loader_rec,
        valid_loader,
        device,
        args,
    ):
        self.model = model_rec
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.metrics = args.metrics.split(",")
        self.generate_num = max([int(m.split("@")[1]) for m in self.metrics])
        self.generate_num = max(self.generate_num, self.args.beam_size)
        self.length_penalty = self.args.length_penalty

        self.train_loader_id = train_loader_id
        self.train_loader_rec = train_loader_rec
        self.model_gen = model_gen  # TODO to be deleted
        self.model_rec = self.model
        self.num_alternations = self.args.rounds
        (
            self.id_optimizer,
            self.id_scheduler,
            self.rec_optimizer,
            self.rec_scheduler,
        ) = self.create_optimizer_and_scheduler()
        self.get_testloader(regenerate=False, phase=0)
        self.cur_model_path = None

        self.best_score = -1
        self.best_epoch = -1
        self.get_validloader(regenerate=False, phase=0)
        self.args = args

    def create_optimizer_and_scheduler(self):
        logging.info("Building Optimizer and Scheduler")
        batch_per_epoch_id = len(self.train_loader_id)
        batch_per_epoch_rec = len(self.train_loader_rec)
        id_total_steps = (
            batch_per_epoch_id
            // self.args.gradient_accumulation_steps
            * self.args.id_epochs
            * self.num_alternations
        )
        id_warmup_steps = int(id_total_steps * self.args.warmup_prop)

        rec_total_steps = (
            batch_per_epoch_rec
            // self.args.gradient_accumulation_steps
            * self.args.rec_epochs
            * self.num_alternations
        )
        rec_warmup_steps = int(rec_total_steps * self.args.warmup_prop)

        logging.info(f"Batch per epoch id: {batch_per_epoch_id}")
        logging.info(f"Warmup proportion: {self.args.warmup_prop}")
        logging.info(f"Total id generator steps: {id_total_steps}")
        logging.info(f"Warm up id generator steps: {id_warmup_steps}")
        logging.info(f"Batch per epoch rec: {batch_per_epoch_rec}")
        logging.info(f"Warmup proportion: {self.args.warmup_prop}")
        logging.info(f"Total rec generator steps: {rec_total_steps}")
        logging.info(f"Warm up rec generator steps: {rec_warmup_steps}")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters_id = [
            {
                "params": [
                    p
                    for n, p in self.model_gen.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model_gen.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_grouped_parameters_rec = [
            {
                "params": [
                    p
                    for n, p in self.model_rec.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model_rec.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        logging.info(f"Building Optimizer AdamW")

        optimizer_id = AdamW(
            optimizer_grouped_parameters_id,
            lr=self.args.id_lr,
            eps=self.args.adam_eps,
        )
        optimizer_rec = AdamW(
            optimizer_grouped_parameters_rec,
            lr=self.args.rec_lr,
            eps=self.args.adam_eps,
        )

        scheduler_id = get_linear_schedule_with_warmup(
            optimizer_id, id_warmup_steps, id_total_steps
        )
        scheduler_rec = get_linear_schedule_with_warmup(
            optimizer_rec, rec_warmup_steps, rec_total_steps
        )

        return optimizer_id, scheduler_id, optimizer_rec, scheduler_rec

    def train_generator(self):
        self.model_gen.zero_grad()
        self.model_rec.zero_grad()
        global_epoch = 0  # save the global training epoch, used for sampler
        train_losses = []
        accumulated_loss = 0.0
        updated = False

        if self.args.alt_style == "id_first":
            raise NotImplementedError
        elif self.args.alt_style == "rec_first":
            for alter in range(self.num_alternations):
                logging.info(f"Training Recommender phase {alter+1}")

                for param in self.model_rec.parameters():
                    param.requires_grad = True
                for param in self.model_gen.parameters():
                    param.requires_grad = False

                for rec_epoch in range(self.args.rec_epochs):
                    self.model_gen.train()
                    self.model_rec.train()
                    logging.info(
                        f"Start training recommender for phase {alter+1}, epoch {rec_epoch+1}"
                    )

                    self.train_loader_rec.sampler.set_epoch(global_epoch)

                    losses = []
                    for step, batch in enumerate(
                        tqdm(self.train_loader_rec, dynamic_ncols=True)
                    ):
                        # print(f">> inside train_generator --- 0"); embed()
                        input_ids = batch["item_text_ids"].to(self.device)
                        attention_mask = batch["item_text_masks"].to(self.device)
                        output_ids = batch["target_ids"].to(self.device)
                        output_masks = batch["target_masks"].to(self.device)

                        loss = self.model_rec(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=output_ids,
                            return_dict=False,
                        )[0]

                        loss = loss / self.args.gradient_accumulation_steps

                        loss.backward()
                        accumulated_loss += loss.item()

                        if (
                            step + 1
                        ) % self.args.gradient_accumulation_steps == 0 or step + 1 == len(
                            self.train_loader_rec
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                self.model_rec.parameters(), self.args.clip
                            )

                            self.rec_optimizer.step()
                            self.rec_scheduler.step()
                            self.model_gen.zero_grad()
                            self.model_rec.zero_grad()

                            losses.append(accumulated_loss)
                            accumulated_loss = 0.0

                    train_epoch_loss = sum(losses) / len(losses)
                    train_losses.append(train_epoch_loss)

                    logging.info(
                        f"The average training loss for rec phase {alter+1} epoch {rec_epoch+1} is {train_epoch_loss}"
                    )

                    if (
                        (rec_epoch + 1) % self.args.save_rec_epochs == 0
                        or rec_epoch + 1 == self.args.rec_epochs
                    ):
                        self.save_model(alter=alter, epoch=rec_epoch, model_type="rec")

                    if (
                        self.args.test_epoch_rec > 0
                        and (rec_epoch + 1) % self.args.test_epoch_rec == 0
                    ):
                        self.model_gen.eval()
                        self.model_rec.eval()

                        if self.args.valid_by_test:
                            self.test()
                        else:
                            self.validate()

                        if updated:
                            self.save_model(
                                alter=alter,
                                epoch=rec_epoch,
                                model_type="rec",
                                is_best=True,
                            )

                    global_epoch += 1

                logging.info(f"Training ID Generator phase {alter+1} finished ..")

            self.save_model(alter=alter, epoch=rec_epoch, model_type="rec")
        else:
            raise NotImplementedError
        return True

    def save_model(self, alter, epoch, model_type, is_best=False):
        if not is_best and model_type == "rec":
            cur_path = os.path.join(
                self.args.model_path, f"model_rec_phase_{alter+1}_epoch_{epoch+1}.pt"
            )
            self.cur_model_path = cur_path

            torch.save(self.model_rec.state_dict(), cur_path)

            cur_path = os.path.join(
                self.args.model_path,
                f"optimizer_rec_phase_{alter+1}_epoch_{epoch+1}.pt",
            )
            torch.save(self.rec_optimizer.state_dict(), cur_path)

            cur_path = os.path.join(
                self.args.model_path,
                f"scheduler_rec_phase_{alter+1}_epoch_{epoch+1}.pt",
            )
            torch.save(self.rec_scheduler.state_dict(), cur_path)

            logging.info(f"Save the current rec model to {cur_path}")

        else:
            cur_path = os.path.join(self.args.model_path, f"model_rec_best.pt")
            self.cur_model_path = cur_path
            torch.save(self.model_rec.state_dict(), cur_path)

            cur_path = os.path.join(self.args.model_path, f"optimizer_rec_best.pt")
            torch.save(self.rec_optimizer.state_dict(), cur_path)

            cur_path = os.path.join(self.args.model_path, f"scheduler_rec_best.pt")
            torch.save(self.rec_scheduler.state_dict(), cur_path)

            logging.info(f"Save the best rec model to {cur_path}")

    def get_testloader(
        self,
        model_gen=None,
        tokenizer=None,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
    ):
        self.testloaders = []
        datasets = self.args.datasets.split(",")  # 'Beauty'
        tasks = self.args.tasks.split(",")  # 'sequential'
        collator = CollatorGRUT(self.tokenizer, args=self.args, mode="test")
        for dataset in datasets:
            for task in tasks:
                testdata = TestDatasetGRUT(
                    self.args,
                    dataset,
                    task,
                    model_gen,
                    tokenizer,
                    regenerate,
                    phase,
                    debug_test_small_set=debug_test_small_set,
                )
                testloader = DataLoader(
                    dataset=testdata,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=collator,
                    shuffle=False,
                )
                self.testloaders.append(testloader)

    def get_validloader(
        self,
        model_gen=None,
        tokenizer=None,
        regenerate=False,
        phase=0,
        debug_test_small_set=False,
    ):
        self.validloaders = []
        datasets = self.args.datasets.split(",")  # 'Beauty'
        tasks = self.args.tasks.split(",")  # 'sequential'
        collator = CollatorGRUT(self.tokenizer, args=self.args, mode="valid")
        for dataset in datasets:
            for task in tasks:
                testdata = TestDatasetGRUT(
                    args=self.args,
                    dataset=dataset,
                    task=task,
                    model_gen=model_gen,
                    tokenizer=tokenizer,
                    regenerate=regenerate,
                    phase=phase,
                    debug_test_small_set=debug_test_small_set,
                    mode="validation",
                )
                testloader = DataLoader(
                    dataset=testdata,
                    batch_size=self.args.eval_batch_size,
                    collate_fn=collator,
                    shuffle=False,
                )
                self.validloaders.append(testloader)

    def test_from_model(self, rec_model_path=None, id_model_path=None):
        self.model.eval()
        if rec_model_path:
            self.model.load_state_dict(
                torch.load(rec_model_path, map_location=self.device), strict=False
            )
        for loader in self.testloaders:
            self.test_dataset_task(loader)

    def test(self, path=None):
        self.get_testloader(
            regenerate=False,
            phase=0,
            debug_test_small_set=self.args.debug_test_small_set,
        )
        self.model.eval()  # self.model ==  self.model_rec
        if path:
            # if path is directory, load from the .pt from the directory
            if os.path.isdir(path):
                path = os.path.join(path, os.listdir(path)[0])
            self.model.load_state_dict(torch.load(path, map_location=self.device))

        if self.args.debug_test_on_train:
            train_loader = get_loader_gram_train(
                args=self.args,
                tokenizer=self.tokenizer,
                TrainSetRec=self.train_loader_rec.dataset,
            )
            train_loader.sampler.set_epoch(0)
            self.debug_test_on_train(train_loader)

        for loader in self.testloaders:
            self.test_dataset_task(loader)

    # Use validation set for testing (for hyperparameter tuning)
    def validate_from_model(self, rec_model_path=None, id_model_path=None):
        self.get_validloader(
            regenerate=False,
            phase=0,
            debug_test_small_set=self.args.debug_test_small_set,
        )
        self.model.eval()
        if rec_model_path:
            self.model.load_state_dict(
                torch.load(rec_model_path, map_location=self.device), strict=False
            )
        for loader in self.validloaders:
            self.test_dataset_task(loader, mode="validation")

    def validate(self, path=None):
        self.get_validloader(
            regenerate=False,
            phase=0,
            debug_test_small_set=self.args.debug_test_small_set,
        )
        self.model.eval()  # self.model ==  self.model_rec
        if path:
            # if path is directory, load from the .pt from the directory
            if os.path.isdir(path):
                path = os.path.join(path, os.listdir(path)[0])
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        for loader in self.validloaders:
            self.test_dataset_task(loader, mode="validation")

    def debug_test_on_train(self, testloader):
        dataset_gram = self.train_loader_rec.dataset.datasets[0]
        logging.info(
            f"[debug] testing {dataset_gram.dataset} dataset on {dataset_gram.tasks[0]} task"
        )

        test_total, cnt = 0, 0
        candidates = dataset_gram.all_items
        if self.args.save_predictions:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("../preds"):
                os.makedirs("../preds")
            pred_fname = f"../preds/{timestamp}_{dataset_gram.dataset}_{dataset_gram.tasks[0]}_pred_debug.tsv"
            pred_file = open(pred_fname, "w")
            pred_file.write(f"idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n")

        with torch.no_grad():
            # candidate: 'rene furterer complexe 5'
            # after encoding: [0] + [3, 1536, 15, 4223, 449, 49, 1561, 15, 305, 1]
            # trie_dict: {0: {3: {1536: {15: {4223: {449: {49: {1561: {15: {305: {1: {}}}}}}}}}}}
            if self.args.item_id_type == "t5_token":
                encoded_candidates = [
                    [0]
                    + self.tokenizer.convert_tokens_to_ids(candidate.split(" "))
                    + [1]
                    for candidate in candidates
                ]
            elif self.args.item_id_type == "split":
                encoded_candidates = []
                for candidate in candidates:
                    tmp_tokens = self.tokenizer.encode(candidate)
                    encoded_candidate = [0]
                    # CAUTION::::: split token hard coded
                    for tok in tmp_tokens:
                        if tok in [1820, 9175]:
                            continue
                        encoded_candidate.append(tok)
                    encoded_candidates.append(encoded_candidate)
            else:
                encoded_candidates = [
                    [0] + self.tokenizer.encode(f"{candidate}")
                    for candidate in candidates
                ]
            candidate_trie = gt.Trie(encoded_candidates)
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

            metrics_res = np.array([0.0] * len(self.metrics))
            logging_str = ""

            for batch in tqdm(testloader, dynamic_ncols=True):

                input_ids = batch["item_text_ids"].to(self.device)
                attention_mask = batch["item_text_masks"].to(self.device)
                output_ids = batch["target_ids"].to(self.device)
                output_masks = batch["target_masks"].to(self.device)

                if self.args.item_id_type == "t5_token":
                    max_length = max(
                        [len(candidate) for candidate in encoded_candidates]
                    )
                elif self.args.item_id_type == "split":
                    max_length = max(
                        [len(candidate) for candidate in encoded_candidates]
                    )
                else:
                    max_length = 50

                prediction = self.model_rec.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=self.generate_num,
                    num_return_sequences=self.generate_num,
                    output_scores=True,
                    return_dict_in_generate=True,
                    length_penalty=self.length_penalty,
                )

                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]

                # Make -100 to 0 of output_ids
                output_ids = torch.where(output_ids == -100, 0, output_ids)
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                rel_results = evaluate.rel_results(
                    generated_sents, gold_sents, prediction_scores, self.generate_num
                )

                test_total += len(rel_results)

                metrics_per_seq = evaluate.get_metrics_results(
                    rel_results, self.metrics
                )
                metrics_res += metrics_per_seq

                if self.args.save_predictions:
                    prediction_scores_tmp = list(
                        prediction_scores.detach().cpu().numpy()
                    )
                    prediction_scores_tmp = [
                        str(score) for score in prediction_scores_tmp
                    ]
                    metrics_per_seq_tmp = [
                        str(score) for score in list(metrics_per_seq)
                    ]
                    user_id = batch["user_ids"][0]
                    write_results = [
                        str(user_id),
                        "\t".join(metrics_per_seq_tmp),
                        gold_sents[0],
                        "||".join(generated_sents),
                        "||".join(prediction_scores_tmp),
                    ]
                    pred_file.write("\t".join(write_results) + "\n")
                    cnt += 1

                if cnt < 10:
                    logging_str += f"[GT] {gold_sents[0]} || [top-1] {generated_sents[0]} || [hits@10] {metrics_per_seq[1]}\n"  ##### TODO hard-coded ['hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']

            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)

            metrics_res /= test_total

            logging.info("\n-----------------Test on train set--------------")
            logging.info(logging_str)

            for i in range(len(self.metrics)):
                logging.info(f"{self.metrics[i]}: {metrics_res[i]}")
                if self.args.save_predictions:
                    pred_file.write(f"{self.metrics[i]}: {metrics_res[i]}\n")
            if self.args.save_predictions:
                pred_file.close()

            logging.info(f">> preds saved to {pred_fname}")

    def get_trend_info(self, testloader, mode="test"):
        """Precompute per-user per-day log_max trend scores.

        Returns:
            trend_score_arr: (num_users, num_items) float ndarray
            trend_score_asin_arr: (num_items,) item-id index -> asin
            trend_score_user_arr: (num_users,) user-id index -> user_id
        """
        trend_reranking_days = self.args.trend_reranking_days
        trend_reranking_use_valid_on_test = self.args.trend_reranking_use_valid_on_test

        dataset_name = testloader.dataset.dataset
        dataset_dir = os.path.join("../rec_datasets", dataset_name)

        train_tsv = os.path.join(dataset_dir, "timestamp_train.tsv")
        valid_tsv = os.path.join(dataset_dir, "timestamp_valid.tsv")
        test_tsv = os.path.join(dataset_dir, "timestamp_test.tsv")

        if os.path.exists(train_tsv):  # Only check train
            train_df = pd.read_csv(train_tsv, sep="\t")
            valid_df = pd.read_csv(valid_tsv, sep="\t")
            test_df = pd.read_csv(test_tsv, sep="\t")
            logging.info(
                f">> Loaded dataframe for reranking (Train {len(train_df)} | Valid {len(valid_df)} | Test {len(test_df)})"
            )
        else:
            # Build timestamp splits by reading rec_datasets/{dataset}/timestamp.txt
            # (in grut-refactoring testloader.dataset does NOT have df_time attribute).
            df_time = pd.read_csv(
                os.path.join(dataset_dir, "timestamp.txt"), sep="\t"
            )
            df_time = df_time.sort_values(["user_id", "timestamp"]).reset_index(
                drop=True
            )

            if dataset_name == "Yelp":  ## Yelp repeat item !!!!
                df_time["item_id"] = df_time["item_id"].str.split(
                    "||", regex=False
                ).str[0]

            train_df, valid_df, test_df = (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )
            for user_id, group in tqdm(df_time.groupby("user_id")):
                test_df = pd.concat([test_df, group.iloc[-1:]])
                valid_df = pd.concat([valid_df, group.iloc[-2:-1]])
                train_df = pd.concat([train_df, group.iloc[:-2]])

            train_df = train_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            train_df.to_csv(train_tsv, sep="\t", index=False)
            valid_df.to_csv(valid_tsv, sep="\t", index=False)
            test_df.to_csv(test_tsv, sep="\t", index=False)

            logging.info(
                f">> Saved dataframe for reranking (Train {len(train_df)} | Valid {len(valid_df)} | Test {len(test_df)})"
            )

        # 1. Trend popularity
        if mode == "test":
            target_df = test_df
            if trend_reranking_use_valid_on_test:  # train + valid as source
                source_df = pd.concat([train_df, valid_df], ignore_index=True)
            else:
                source_df = train_df
        else:
            target_df = valid_df
            source_df = train_df

        target_df["date"] = pd.to_datetime(target_df["datetime"]).dt.date
        source_df["date"] = pd.to_datetime(source_df["datetime"]).dt.date

        # Precompute daily item counts
        daily_item_counts = (
            source_df.groupby(["date", "item_id"]).size().reset_index(name="count")
        )
        daily_item_counts.sort_values(by="date", inplace=True)
        all_dates = pd.date_range(
            start=source_df["date"].min(), end=target_df["date"].max()
        )

        # Fill missing dates (row: date, col: item_id, value: count -> cumulative count per item up to date)
        daily_pivot = daily_item_counts.pivot(
            index="date", columns="item_id", values="count"
        ).fillna(0)
        daily_pivot = daily_pivot.reindex(all_dates, fill_value=0)
        daily_cumsum = daily_pivot.cumsum()

        # Cache file names: top0 is a frozen placeholder (top_k_items not used)
        if not trend_reranking_use_valid_on_test:
            fname = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_log_max.npy",
            )
            fname2 = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_asin.pkl",
            )
            fname3 = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_user_pkl",
            )
        else:
            fname = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_log_max_use_valid.npy",
            )
            fname2 = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_asin_use_valid.pkl",
            )
            fname3 = os.path.join(
                dataset_dir,
                f"timestamp_day{trend_reranking_days}_score_user_use_valid_pkl",
            )

        trend_score_arr, trend_score_asin_arr, trend_score_user_arr = None, None, None
        if (
            os.path.exists(fname)
            and os.path.exists(fname2)
            and os.path.exists(fname3)
        ):
            trend_score_arr = np.load(fname)
            logging.info(
                f">> Loaded trend score for latest ({trend_reranking_days} days ({len(trend_score_arr)})"
            )

            trend_score_asin_arr = pickle.load(open(fname2, "rb"))
            logging.info(
                f">> Loaded trend score for latest ({trend_reranking_days} days idx -> asin) ({len(trend_score_asin_arr)})"
            )

            trend_score_user_arr = pickle.load(open(fname3, "rb"))
            logging.info(
                f">> Loaded trend score for latest ({trend_reranking_days} days idx -> user_id) ({len(trend_score_user_arr)})"
            )
        else:
            trend_scores_list = []
            user_ids_list = []

            for _, row in tqdm(
                target_df.iterrows(),
                total=len(target_df),
                dynamic_ncols=True,
                desc="Precompute trend score",
            ):
                user_id = row["user_id"]
                cur_date = pd.to_datetime(row["date"])

                date_end = cur_date - pd.Timedelta(days=1)  # exclude current day
                date_start = cur_date - pd.Timedelta(days=trend_reranking_days)

                # exception if out of range
                if date_end < daily_cumsum.index[0]:
                    cnts = pd.Series(0, index=daily_cumsum.columns)
                elif date_start <= daily_cumsum.index[0]:
                    cnts = daily_cumsum.loc[date_end]
                else:
                    cnts = daily_cumsum.loc[date_end] - daily_cumsum.loc[date_start]

                item_cnts = cnts.values
                if trend_score_asin_arr is None:
                    trend_score_asin_arr = np.array(cnts.index.tolist())

                trend_scores = log_max_trend_score(item_cnts)

                trend_scores_list.append(trend_scores)
                user_ids_list.append(user_id)

            trend_score_arr = np.vstack(trend_scores_list)
            np.save(fname, trend_score_arr)
            logging.info(
                f">> Saved trend score for latest ({trend_reranking_days} days) ({len(trend_score_arr)})"
            )

            with open(fname2, "wb") as outf:
                pickle.dump(trend_score_asin_arr, outf)
                logging.info(
                    f">> Saved trend score for latest ({trend_reranking_days} days) (idx -> asin) ({len(trend_score_asin_arr)})"
                )

            with open(fname3, "wb") as outf:
                trend_score_user_arr = np.array(user_ids_list)
                pickle.dump(trend_score_user_arr, outf)
                logging.info(
                    f">> Saved trend score for latest ({trend_reranking_days} days) (user_id) ({len(user_ids_list)})"
                )

        return trend_score_arr, trend_score_asin_arr, trend_score_user_arr

    def test_dataset_task(self, testloader, mode="test"):
        logging.info(
            f"[{mode}] testing {testloader.dataset.dataset} dataset on {testloader.dataset.task} task"
        )

        test_total, cnt = 0, 0
        candidates = testloader.dataset.all_items

        total_time = 0

        if self.args.save_predictions:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists("../preds"):
                os.makedirs("../preds")
            pred_fname = f"../preds/{timestamp}_{testloader.dataset.dataset}_{testloader.dataset.task}_pred_{mode}.tsv"
            pred_file = open(pred_fname, "w")
            pred_file.write(f"idx\tH@5\tH@10\tNDCG@5\tNDCG@10\tgold\tpred\tscores\n")

        with torch.no_grad():
            # candidate: 'rene furterer complexe 5'
            # after encoding: [0] + [3, 1536, 15, 4223, 449, 49, 1561, 15, 305, 1]
            # trie_dict: {0: {3: {1536: {15: {4223: {449: {49: {1561: {15: {305: {1: {}}}}}}}}}}}
            if self.args.item_id_type == "t5_token":
                encoded_candidates = [
                    [0]
                    + self.tokenizer.convert_tokens_to_ids(candidate.split(" "))
                    + [1]
                    for candidate in candidates
                ]
            elif self.args.item_id_type == "split":
                encoded_candidates = []
                for candidate in candidates:
                    tmp_tokens = self.tokenizer.encode(candidate)
                    encoded_candidate = [0]
                    # CAUTION::::: split token hard coded
                    for tok in tmp_tokens:
                        if tok in [1820, 9175]:
                            continue
                        encoded_candidate.append(tok)
                    encoded_candidates.append(encoded_candidate)
            else:
                encoded_candidates = [
                    [0] + self.tokenizer.encode(f"{candidate}")
                    for candidate in candidates
                ]
            candidate_trie = gt.Trie(encoded_candidates)

            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)

            # Trend-aware inference rerank setup (score_adjust method).
            # λ=0 bypasses both precomputation and per-batch rerank.
            trend_reranking_lambda = self.args.trend_reranking_lambda
            if trend_reranking_lambda > 0:
                (
                    trend_score_arr,
                    trend_score_asin_arr,
                    trend_score_user_arr,
                ) = self.get_trend_info(testloader, mode=mode)

                # Build mappings to convert generated lexids back to item asins,
                # then to trend index.
                lexid2item = {
                    v: k for k, v in testloader.dataset.item2lexid.items()
                }
                item2tokens = {}
                for lexid, tokens in zip(candidates, encoded_candidates):
                    item2tokens[lexid2item[lexid]] = tokens
                item2decodedtokens = {
                    k: self.tokenizer.decode(v, skip_special_tokens=True)
                    for k, v in item2tokens.items()
                }
                decodedtokens2item = {v: k for k, v in item2decodedtokens.items()}

                itemasin_to_index = {
                    asin: idx for idx, asin in enumerate(trend_score_asin_arr)
                }
                user_to_index = {
                    user: idx for idx, user in enumerate(trend_score_user_arr)
                }

            # Predictions cache: first run does beam search and writes to disk;
            # subsequent λ-tune runs load from cache and skip beam search.
            if self.args.rec_model_path:
                predictions_cache_dir = '/'.join(self.args.rec_model_path.split('/')[:-1])
            else:
                predictions_cache_dir = self.args.full_log_dir
            predictions_cache_file = os.path.join(
                predictions_cache_dir,
                f'predictions_all_{mode}_beam{self.args.beam_size}_len{len(testloader)}.pkl'
            )

            do_write_preds_cache = False
            if os.path.exists(predictions_cache_file):
                with open(predictions_cache_file, 'rb') as f:
                    predictions_all = pickle.load(f)
                print(f"[INFO] Loaded cached predictions (beam {self.args.beam_size}) from {predictions_cache_file}")
            else:
                do_write_preds_cache = True
                print(f"[INFO] No cache found. Generating predictions and saving to {predictions_cache_file}")
                predictions_all = []

            metrics_res = np.array([0.0] * len(self.metrics))
            logging_str = ""
            for batch_idx, batch in enumerate(tqdm(testloader, dynamic_ncols=True)):

                input_ids = batch["item_text_ids"].to(self.device)
                attention_mask = batch["item_text_masks"].to(self.device)
                output_ids = batch["target_ids"].to(self.device)
                output_masks = batch["target_masks"].to(self.device)

                if self.args.item_id_type == "t5_token":
                    max_length = max(
                        [len(candidate) for candidate in encoded_candidates]
                    )
                elif self.args.item_id_type == "split":
                    max_length = max(
                        [len(candidate) for candidate in encoded_candidates]
                    )
                else:
                    max_length = 50

                if do_write_preds_cache:
                    start_time = time()
                    prediction = self.model_rec.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                        length_penalty=self.length_penalty,
                    )
                    total_time += time() - start_time
                    prediction_ids = prediction["sequences"]
                    prediction_scores = prediction["sequences_scores"]
                    predictions_all.append((prediction_ids.cpu(), prediction_scores.cpu()))
                else:
                    prediction_ids, prediction_scores = predictions_all[batch_idx]
                    prediction_ids = prediction_ids.to(self.device)
                    prediction_scores = prediction_scores.to(self.device)

                # Trend-aware inference rerank (score_adjust):
                # final_scores = original_scores + λ * trend_scores, then re-sort.
                if trend_reranking_lambda > 0:
                    user_id = batch["user_ids"][0]
                    prediction_ids_orig = prediction_ids
                    prediction_scores_orig = prediction_scores

                    generated_sents_for_trend = self.tokenizer.batch_decode(
                        prediction_ids_orig, skip_special_tokens=True
                    )
                    generated_asins = [
                        decodedtokens2item[sent]
                        for sent in generated_sents_for_trend
                    ]
                    indices = [
                        itemasin_to_index.get(asin, -1) for asin in generated_asins
                    ]
                    trend_scores = trend_score_arr[
                        user_to_index[user_id], indices
                    ]

                    if not isinstance(prediction_scores_orig, np.ndarray):
                        prediction_scores_tmp = list(
                            prediction_scores_orig.detach().cpu().numpy()
                        )
                    else:
                        prediction_scores_tmp = list(prediction_scores_orig)

                    final_scores = (
                        np.array(prediction_scores_tmp)
                        + trend_reranking_lambda * trend_scores
                    )
                    prediction_scores = torch.tensor(final_scores).to(self.device)
                    prediction_scores, sort_indices = torch.sort(
                        prediction_scores, descending=True
                    )
                    prediction_ids = prediction_ids_orig[sort_indices]

                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )

                rel_results = evaluate.rel_results(
                    generated_sents, gold_sents, prediction_scores, self.generate_num
                )

                test_total += len(rel_results)

                metrics_per_seq = evaluate.get_metrics_results(
                    rel_results, self.metrics
                )
                metrics_res += metrics_per_seq

                if self.args.save_predictions:
                    prediction_scores_tmp = list(
                        prediction_scores.detach().cpu().numpy()
                    )
                    prediction_scores_tmp = [
                        str(score) for score in prediction_scores_tmp
                    ]
                    metrics_per_seq_tmp = [
                        str(score) for score in list(metrics_per_seq)
                    ]
                    user_id = batch["user_ids"][0]
                    write_results = [
                        str(user_id),
                        "\t".join(metrics_per_seq_tmp),
                        gold_sents[0],
                        "||".join(generated_sents),
                        "||".join(prediction_scores_tmp),
                    ]
                    pred_file.write("\t".join(write_results) + "\n")
                    cnt += 1

                if cnt < 10:
                    logging_str += f"[GT] {gold_sents[0]} || [top-1] {generated_sents[0]} || [hits@10] {metrics_per_seq[1]}\n"  ##### TODO hard-coded ['hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']

            if do_write_preds_cache:
                with open(predictions_cache_file, 'wb') as f:
                    pickle.dump(predictions_all, f)
                print(f"[INFO] Saved predictions cache to {predictions_cache_file}")

            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)

            metrics_res /= test_total

            logging.info("\n-------------------------------")
            logging.info(logging_str)

            for i in range(len(self.metrics)):
                logging.info(f"{mode} {self.metrics[i]}: {metrics_res[i]}")
                if self.args.save_predictions:
                    pred_file.write(f"{self.metrics[i]}: {metrics_res[i]}\n")

            logging.info(
                f"Total inference time: {total_time:.2f}s for {len(testloader)} samples. Average: {total_time/len(testloader):.4f}s"
            )

            if self.args.save_predictions:
                pred_file.close()

            logging.info(f">> preds saved to {pred_fname}")
