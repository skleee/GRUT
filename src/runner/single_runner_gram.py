import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from time import time

import utils.generation_trie as gt
import utils.evaluate as evaluate

from data import TestDatasetGRAM
from utils import get_loader_gram_train
from processor import CollatorGRAM


class SingleRunnerGRAM:
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
        collator = CollatorGRAM(self.tokenizer, args=self.args, mode="test")
        for dataset in datasets:
            for task in tasks:
                testdata = TestDatasetGRAM(
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
        collator = CollatorGRAM(self.tokenizer, args=self.args, mode="valid")
        for dataset in datasets:
            for task in tasks:
                testdata = TestDatasetGRAM(
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
