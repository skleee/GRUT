import torch


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text["input"] for input_text in batch]
        output_texts = [input_text["output"] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


class TestCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        user_idx = [input_text["user_idx"] for input_text in batch]
        input_texts = [input_text["input"] for input_text in batch]
        output_texts = [input_text["output"] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
            torch.tensor(user_idx),
        )


class CollatorGen:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output_texts = [input_text["output_prompt"] for input_text in batch]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts,
            padding="longest",
            truncation=True,
            max_length=512,  # [SK] CAUTION !!! MAX LENGTH HARD CODED
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]
        histories = [input_text["history"] for input_text in batch]
        input_prompt = [input_text["input_prompt"] for input_text in batch]
        hist_lengths = [len(hist) for hist in histories]

        # add placeholders to the input prompt
        input_prompt_ph = []  # with placeholders
        input_insert_positions = []
        tokenized_prompts = []  # Store tokenized prompts here
        for i, p in enumerate(input_prompt):
            length = hist_lengths[i]
            p_s = p.replace("{history}", " ; " * (length))
            tokens = self.tokenizer.tokenize(p_s)
            insert_p = [1 if token == ";" else 0 for token in tokens]
            tokenized_prompts.append(tokens)
            input_prompt_ph.append(p_s)
            input_insert_positions.append(insert_p)

        # process input prompt
        input_prompt_inputs = self.tokenizer.batch_encode_plus(
            tokenized_prompts,
            is_split_into_words=True,
            padding="longest",
            truncation=True,
            max_length=512,
        )

        # pad input prompt insert positions
        input_prompt_len = len(input_prompt_inputs["input_ids"][0])
        for insert_p in input_insert_positions:
            while len(insert_p) < input_prompt_len:
                insert_p.append(0)

        # process history
        flattened_histories = [plain_text for hist in histories for plain_text in hist]

        # process input history, need two level of paddings, history level and plain text level
        history_inputs = self.tokenizer.batch_encode_plus(
            flattened_histories, padding="longest", truncation=True, max_length=256
        )
        max_hist_token = len(history_inputs["input_ids"][0])
        hist_lengths = [len(hist) for hist in histories]

        # Apply padding at the history level
        padded_histories = []
        padded_attention_mask_histories = []
        max_hist_length = max(hist_lengths)
        current_index = 0

        for length in hist_lengths:
            padded_hist = torch.zeros(
                (max_hist_length, max_hist_token), dtype=torch.long
            )
            padded_attention_mask = torch.zeros((max_hist_length, max_hist_token))
            padded_hist[:length] = torch.tensor(
                history_inputs["input_ids"][current_index : current_index + length],
                dtype=torch.long,
            )
            padded_attention_mask[:length] = torch.tensor(
                history_inputs["attention_mask"][current_index : current_index + length]
            )
            padded_histories.append(padded_hist)
            padded_attention_mask_histories.append(padded_attention_mask)
            current_index += length
        history_input_ids = torch.stack(padded_histories)
        history_input_attention = torch.stack(padded_attention_mask_histories)

        return (
            torch.tensor(input_prompt_inputs["input_ids"]),
            torch.tensor(input_insert_positions),
            # torch.tensor(output_prompt_inputs['input_ids']),
            # torch.tensor(output_insert_positions),
            history_input_ids,
            history_input_attention,
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


class CollatorGRAM:
    def __init__(self, tokenizer, args=None, mode="train"):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.item_prompt_max_len = (
            args.item_prompt_max_len
        )  # maximum length of item text
        self.target_max_len = args.target_max_len  # maximum length of target text
        self.max_item_num = (
            args.max_his
        )  # maximum number of items in a single sequence (N)
        self.item_id_type = args.item_id_type
        self.hierarchical_id_type = args.hierarchical_id_type

    def __call__(self, batch):
        input_texts = [input_text["input"] for input_text in batch]
        output_texts = [input_text["output"] for input_text in batch]
        if self.item_id_type == "t5_token":
            output_ids = [
                self.tokenizer.convert_tokens_to_ids(text.split(" ")) + [1]
                for text in output_texts
            ]
            max_len = max([len(ids) for ids in output_ids])
            target = {
                "input_ids": torch.tensor(
                    [ids + [0] * (max_len - len(ids)) for ids in output_ids]
                ),
                "attention_mask": torch.tensor(
                    [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in output_ids]
                ),
            }
        elif (
            self.item_id_type == "split"
        ):  # id :: |▁butter|▁mango|generation|▁lend|▁obtained|▁said|▁kernel
            target = self.encode_target_split(output_texts)
        else:
            target = self.tokenizer.batch_encode_plus(
                output_texts,
                max_length=self.target_max_len if self.target_max_len > 0 else None,
                padding="longest",
                return_tensors="pt",
                truncation=True if self.target_max_len > 0 else False,
            )
        target_ids = target["input_ids"]
        target_masks = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_masks, -100)

        if self.item_id_type == "split":
            item_text_ids, item_text_masks = self.encode_texts_split(
                input_texts, self.tokenizer
            )
        else:
            item_text_ids, item_text_masks = self.encode_texts(
                input_texts, self.tokenizer
            )

        neg_item_ids, neg_item_masks = None, None

        # add user id to the input ('A2CG5Y82ZZNY6W')
        user_ids = [batch_item["user_id"] for batch_item in batch]

        return {
            "target_ids": target_ids,  # B x L
            "target_masks": target_masks,  # B x L
            "item_text_ids": item_text_ids,  # B x (N + 1) x L
            "item_text_masks": item_text_masks,  # B x (N + 1) x L
            "neg_item_ids": neg_item_ids,  # B x M x L
            "neg_item_masks": neg_item_masks,  # B x M x L
            "user_ids": user_ids,
        }

    def encode_texts(self, batch_item_texts, tokenizer):
        item_text_ids, item_text_masks = [], []

        max_item_batch = max([len(input_text) for input_text in batch_item_texts])
        max_item_num = min(max_item_batch, self.max_item_num)
        max_item_num = max_item_num + 1  # one for coarse-grained user prompt

        for _, text_passages in enumerate(batch_item_texts):
            p = tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.item_prompt_max_len,
                pad_to_max_length=True,
                return_tensors="pt",
                truncation=True,
            )

            # add additional padding to make all sequences have the same length
            if len(text_passages) < max_item_num:
                p["input_ids"] = torch.cat(
                    [
                        p["input_ids"],
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_prompt_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
                p["attention_mask"] = torch.cat(
                    [
                        p["attention_mask"],
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_prompt_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )

            item_text_ids.append(p["input_ids"][None])  # [None]: add a new dimension
            item_text_masks.append(p["attention_mask"][None])

        item_text_ids = torch.cat(item_text_ids, dim=0)
        item_text_masks = torch.cat(item_text_masks, dim=0)

        max_item_text_len = item_text_masks.sum(-1).max().item()
        item_text_ids = item_text_ids[:, :, :max_item_text_len]
        item_text_masks = item_text_masks[:, :, :max_item_text_len]

        return item_text_ids, item_text_masks.bool()

    def encode_target_split(self, batch_output_texts):
        target = self.tokenizer.batch_encode_plus(
            batch_output_texts,
            max_length=99,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = target["input_ids"]
        attention_mask = target["attention_mask"]

        max_len = self.target_max_len

        filtered_input_ids = []
        filtered_attention_mask = []

        mask = (input_ids != 1820) & (input_ids != 9175)  # '|', '_|'
        indices = torch.nonzero(mask, as_tuple=False)
        for i in range(target["input_ids"].size(0)):
            valid_indices = indices[indices[:, 0] == i, 1]

            # filter out invalid tokens
            tmp_input_ids = input_ids[i, valid_indices][:max_len]

            # check if eos token exists
            if 1 not in tmp_input_ids:
                tmp_input_ids[-1] = 1

            if 1 in tmp_input_ids and tmp_input_ids.size(-1) < max_len:
                tmp_input_ids = torch.cat(
                    [
                        tmp_input_ids,
                        torch.zeros(max_len - tmp_input_ids.size(-1), dtype=torch.long),
                    ]
                )

            tmp_attention_mask = attention_mask[i, valid_indices][:max_len]
            if tmp_attention_mask.size(-1) < max_len:
                tmp_attention_mask = torch.cat(
                    [
                        tmp_attention_mask,
                        torch.zeros(
                            max_len - tmp_attention_mask.size(-1), dtype=torch.long
                        ),
                    ]
                )

            filtered_input_ids.append(tmp_input_ids)
            filtered_attention_mask.append(tmp_attention_mask)
        filtered_input_ids = torch.stack(filtered_input_ids)
        filtered_attention_mask = torch.stack(filtered_attention_mask)

        max_target_len = filtered_attention_mask.sum(-1).max().item()
        filtered_input_ids = filtered_input_ids[:, :max_target_len]
        filtered_attention_mask = filtered_attention_mask[:, :max_target_len]

        return {
            "input_ids": filtered_input_ids,
            "attention_mask": filtered_attention_mask,
        }

    def encode_texts_split(self, batch_item_texts, tokenizer):
        # CAUTION:: Always assume user prompt: 'What would user purchase after {} ; {} ?'
        # CAUTION:: Always assume item prompt: 'item: {}, similar items: {} ; {}'

        item_text_ids, item_text_masks = [], []

        max_item_batch = max([len(input_text) for input_text in batch_item_texts])
        max_item_num = min(max_item_batch, self.max_item_num)
        max_item_num = max_item_num + 1  # one for coarse-grained user prompt

        for bidx, text_passages in enumerate(batch_item_texts):
            p = tokenizer.batch_encode_plus(
                text_passages,
                max_length=999,  # self.item_prompt_max_len,
                pad_to_max_length=True,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = p["input_ids"]
            attention_mask = p["attention_mask"]

            max_len = self.item_prompt_max_len

            # CAUTION::::: split token hard coded
            mask = (input_ids != 1820) & (input_ids != 9175)  # '|', '_|'
            valid_lens = mask.sum(dim=1)
            indices = torch.nonzero(mask, as_tuple=False)

            # filter |, _|
            filtered_input_ids, filtered_attention_mask = [], []
            for i in range(input_ids.size(0)):
                valid_indices = indices[indices[:, 0] == i, 1]

                # filter out invalid tokens
                tmp_input_ids = input_ids[i, valid_indices][:max_len]

                # check if eos token exists
                if 1 not in tmp_input_ids:
                    tmp_input_ids[-1] = 1

                if 1 in tmp_input_ids and tmp_input_ids.size(-1) < max_len:
                    tmp_input_ids = torch.cat(
                        [
                            tmp_input_ids,
                            torch.zeros(
                                max_len - tmp_input_ids.size(-1), dtype=torch.long
                            ),
                        ]
                    )

                tmp_attention_mask = attention_mask[i, valid_indices][:max_len]
                if tmp_attention_mask.size(-1) < max_len:
                    tmp_attention_mask = torch.cat(
                        [
                            tmp_attention_mask,
                            torch.zeros(
                                max_len - tmp_attention_mask.size(-1), dtype=torch.long
                            ),
                        ]
                    )

                filtered_input_ids.append(tmp_input_ids)
                filtered_attention_mask.append(tmp_attention_mask)

            filtered_input_ids = torch.stack(filtered_input_ids)
            filtered_attention_mask = torch.stack(filtered_attention_mask)

            # add additional padding to make all sequences have the same length
            if len(text_passages) < max_item_num:
                filtered_input_ids = torch.cat(
                    [
                        filtered_input_ids,
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_prompt_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
                filtered_attention_mask = torch.cat(
                    [
                        filtered_attention_mask,
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_prompt_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )

            item_text_ids.append(
                filtered_input_ids[None]
            )  # [None]: add a new dimension
            item_text_masks.append(filtered_attention_mask[None])

        item_text_ids = torch.cat(item_text_ids, dim=0)
        item_text_masks = torch.cat(item_text_masks, dim=0)

        max_item_text_len = item_text_masks.sum(-1).max().item()
        item_text_ids = item_text_ids[:, :, :max_item_text_len]
        item_text_masks = item_text_masks[:, :, :max_item_text_len]

        return item_text_ids, item_text_masks.bool()


class CollatorGRUT:
    """GRUT-specific collator.

    Batch layout: each sample's `input` is a list of `1 + global_context_last_N` text passages:
        slot 0       = C_u prompt sentence (user-level temporal context)
        slot 1..L    = C_v lines (item-level transition context)
    The collator tokenizes each passage to length `item_text_max_len` (pad/truncate)
    and stacks to shape `(B, max_item_num, item_text_max_len)`.
    Target is encoded via the same `split`-token filtering used by CollatorGRAM.
    """

    def __init__(self, tokenizer, args=None, mode="train"):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        # Prefer GRUT's item_text_max_len; fall back to GRAM's item_prompt_max_len.
        self.item_text_max_len = getattr(args, "item_text_max_len", None) or args.item_prompt_max_len
        self.target_max_len = args.target_max_len
        # FiD slot count: 1 C_u + L C_v slots.
        self.max_item_num = 1 + args.global_context_last_N
        self.item_id_type = args.item_id_type
        self.hierarchical_id_type = args.hierarchical_id_type
        self.prompt_sentence_for_fid_input = args.prompt_sentence_for_fid_input
        self.global_context = args.global_context
        self.global_context_last_N = args.global_context_last_N
        self.global_context_top_K = args.global_context_top_K

    def __call__(self, batch):
        input_texts = [b["input"] for b in batch]
        output_texts = [b["output"] for b in batch]

        if self.item_id_type == "split":
            target = self.encode_target_split(output_texts)
            item_text_ids, item_text_masks = self.encode_texts_split(input_texts, self.tokenizer)
        else:
            # Non-paper-final fallback — encode targets/texts via the default path.
            target = self.tokenizer.batch_encode_plus(
                output_texts,
                max_length=self.target_max_len if self.target_max_len > 0 else None,
                padding="longest",
                return_tensors="pt",
                truncation=True if self.target_max_len > 0 else False,
            )
            item_text_ids, item_text_masks = self.encode_texts(input_texts, self.tokenizer)

        target_ids = target["input_ids"]
        target_masks = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_masks, -100)

        user_ids = [b["user_id"] for b in batch]

        return {
            "target_ids": target_ids,            # B x L_t
            "target_masks": target_masks,        # B x L_t
            "item_text_ids": item_text_ids,      # B x (1+L) x L_x
            "item_text_masks": item_text_masks,  # B x (1+L) x L_x
            "neg_item_ids": None,
            "neg_item_masks": None,
            "user_ids": user_ids,
        }

    def encode_texts(self, batch_item_texts, tokenizer):
        item_text_ids, item_text_masks = [], []

        max_item_batch = max([len(input_text) for input_text in batch_item_texts])
        max_item_num = min(max_item_batch, self.max_item_num)
        # NOTE: No +1 here. GRUT's self.max_item_num is already 1+L (C_u slot included).
        # GRAM added +1 for its coarse-grained user prompt; GRUT's slot 0 is already C_u.

        for _, text_passages in enumerate(batch_item_texts):
            p = tokenizer.batch_encode_plus(
                text_passages,
                max_length=self.item_text_max_len,
                pad_to_max_length=True,
                return_tensors="pt",
                truncation=True,
            )

            # add additional padding to make all sequences have the same length
            if len(text_passages) < max_item_num:
                p["input_ids"] = torch.cat(
                    [
                        p["input_ids"],
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_text_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
                p["attention_mask"] = torch.cat(
                    [
                        p["attention_mask"],
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_text_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )

            item_text_ids.append(p["input_ids"][None])  # [None]: add a new dimension
            item_text_masks.append(p["attention_mask"][None])

        item_text_ids = torch.cat(item_text_ids, dim=0)
        item_text_masks = torch.cat(item_text_masks, dim=0)

        max_item_text_len = item_text_masks.sum(-1).max().item()
        item_text_ids = item_text_ids[:, :, :max_item_text_len]
        item_text_masks = item_text_masks[:, :, :max_item_text_len]

        return item_text_ids, item_text_masks.bool()

    def encode_target_split(self, batch_output_texts):
        target = self.tokenizer.batch_encode_plus(
            batch_output_texts,
            max_length=99,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = target["input_ids"]
        attention_mask = target["attention_mask"]

        max_len = self.target_max_len

        filtered_input_ids = []
        filtered_attention_mask = []

        mask = (input_ids != 1820) & (input_ids != 9175)  # '|', '_|'
        indices = torch.nonzero(mask, as_tuple=False)
        for i in range(target["input_ids"].size(0)):
            valid_indices = indices[indices[:, 0] == i, 1]

            # filter out invalid tokens
            tmp_input_ids = input_ids[i, valid_indices][:max_len]

            # check if eos token exists
            if 1 not in tmp_input_ids:
                tmp_input_ids[-1] = 1

            if 1 in tmp_input_ids and tmp_input_ids.size(-1) < max_len:
                tmp_input_ids = torch.cat(
                    [
                        tmp_input_ids,
                        torch.zeros(max_len - tmp_input_ids.size(-1), dtype=torch.long),
                    ]
                )

            tmp_attention_mask = attention_mask[i, valid_indices][:max_len]
            if tmp_attention_mask.size(-1) < max_len:
                tmp_attention_mask = torch.cat(
                    [
                        tmp_attention_mask,
                        torch.zeros(
                            max_len - tmp_attention_mask.size(-1), dtype=torch.long
                        ),
                    ]
                )

            filtered_input_ids.append(tmp_input_ids)
            filtered_attention_mask.append(tmp_attention_mask)
        filtered_input_ids = torch.stack(filtered_input_ids)
        filtered_attention_mask = torch.stack(filtered_attention_mask)

        max_target_len = filtered_attention_mask.sum(-1).max().item()
        filtered_input_ids = filtered_input_ids[:, :max_target_len]
        filtered_attention_mask = filtered_attention_mask[:, :max_target_len]

        return {
            "input_ids": filtered_input_ids,
            "attention_mask": filtered_attention_mask,
        }

    def encode_texts_split(self, batch_item_texts, tokenizer):
        # CAUTION:: Always assume user prompt: 'What would user purchase after {} ; {} ?'
        # CAUTION:: Always assume item prompt: 'item: {}, similar items: {} ; {}'

        item_text_ids, item_text_masks = [], []

        max_item_batch = max([len(input_text) for input_text in batch_item_texts])
        max_item_num = min(max_item_batch, self.max_item_num)
        # NOTE: No +1 here. GRUT's self.max_item_num is already 1+L (C_u slot included).
        # GRAM added +1 for its coarse-grained user prompt; GRUT's slot 0 is already C_u.

        for bidx, text_passages in enumerate(batch_item_texts):
            p = tokenizer.batch_encode_plus(
                text_passages,
                max_length=999,  # self.item_text_max_len,
                pad_to_max_length=True,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = p["input_ids"]
            attention_mask = p["attention_mask"]

            max_len = self.item_text_max_len

            # CAUTION::::: split token hard coded
            mask = (input_ids != 1820) & (input_ids != 9175)  # '|', '_|'
            valid_lens = mask.sum(dim=1)
            indices = torch.nonzero(mask, as_tuple=False)

            # filter |, _|
            filtered_input_ids, filtered_attention_mask = [], []
            for i in range(input_ids.size(0)):
                valid_indices = indices[indices[:, 0] == i, 1]

                # filter out invalid tokens
                tmp_input_ids = input_ids[i, valid_indices][:max_len]

                # check if eos token exists
                if 1 not in tmp_input_ids:
                    tmp_input_ids[-1] = 1

                if 1 in tmp_input_ids and tmp_input_ids.size(-1) < max_len:
                    tmp_input_ids = torch.cat(
                        [
                            tmp_input_ids,
                            torch.zeros(
                                max_len - tmp_input_ids.size(-1), dtype=torch.long
                            ),
                        ]
                    )

                tmp_attention_mask = attention_mask[i, valid_indices][:max_len]
                if tmp_attention_mask.size(-1) < max_len:
                    tmp_attention_mask = torch.cat(
                        [
                            tmp_attention_mask,
                            torch.zeros(
                                max_len - tmp_attention_mask.size(-1), dtype=torch.long
                            ),
                        ]
                    )

                filtered_input_ids.append(tmp_input_ids)
                filtered_attention_mask.append(tmp_attention_mask)

            filtered_input_ids = torch.stack(filtered_input_ids)
            filtered_attention_mask = torch.stack(filtered_attention_mask)

            # add additional padding to make all sequences have the same length
            if len(text_passages) < max_item_num:
                filtered_input_ids = torch.cat(
                    [
                        filtered_input_ids,
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_text_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
                filtered_attention_mask = torch.cat(
                    [
                        filtered_attention_mask,
                        torch.zeros(
                            (
                                max_item_num - len(text_passages),
                                self.item_text_max_len,
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )

            item_text_ids.append(
                filtered_input_ids[None]
            )  # [None]: add a new dimension
            item_text_masks.append(filtered_attention_mask[None])

        item_text_ids = torch.cat(item_text_ids, dim=0)
        item_text_masks = torch.cat(item_text_masks, dim=0)

        max_item_text_len = item_text_masks.sum(-1).max().item()
        item_text_ids = item_text_ids[:, :, :max_item_text_len]
        item_text_masks = item_text_masks[:, :, :max_item_text_len]

        return item_text_ids, item_text_masks.bool()
