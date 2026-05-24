import os
import re
import copy

from tqdm import tqdm
from utils import utils


def generative_indexing_id(
    data_path,
    dataset,
    user_sequence_dict,
    phase=0,
    user_id_without_target_item=False,
    args=None,
):
    """
    Use generative indexing method to index the given user seuqnece dict.
    """
    item_text_file = os.path.join(data_path, dataset, "item_plain_text.txt")

    user_sequence_file = os.path.join(data_path, dataset, "user_sequence.txt")

    item_info = utils.ReadLineFromFile(item_text_file)
    item_map = get_dict_from_lines(item_info)

    user_index_file = os.path.join(
        data_path, dataset, f"user_generative_index_phase_{phase}.txt"
    )
    if (
        user_id_without_target_item
    ):  # without data leakage (exclude last item from generating  user id)
        user_index_file = os.path.join(
            data_path, dataset, f"user_generative_index_phase_{phase}_no_last_item.txt"
        )
    else:  # original
        user_index_file = os.path.join(
            data_path, dataset, f"user_generative_index_phase_{phase}.txt"
        )
    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)

    user_sequence_info = utils.ReadLineFromFile(user_sequence_file)
    user_sequence = get_dict_from_lines(user_sequence_info)

    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    return reindex_user_sequence_dict, item_map


def generative_indexing_rec(
    data_path,
    dataset,
    user_sequence_dict,
    model_gen,
    tokenizer,
    regenerate=True,
    phase=0,
    return_user_map=False,
    user_id_without_target_item=False,
    args=None,
):
    """
    Use generative indexing method to index the given user seuqnece dict.
    Generate ID and save to local first
    regenerate: if regenerate id file
    """
    # user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_text_file = os.path.join(data_path, dataset, "item_plain_text.txt")
    user_sequence_file = os.path.join(data_path, dataset, "user_sequence.txt")

    # generate item id file
    if (
        args is not None
        and len(args.item_id_path)
        and os.path.exists(os.path.join(data_path, dataset, args.item_id_path))
    ):
        item_index_file = os.path.join(data_path, dataset, args.item_id_path)
        print(f" ################## Load item id from {item_index_file}")
    else:
        item_index_file = os.path.join(
            data_path, dataset, f"item_generative_indexing_phase_{phase}.txt"
        )
        if (phase == 0 and not os.path.exists(item_index_file)) or (
            phase != 0 and regenerate
        ):
            print(f"(re)generate textual id with id generator phase {phase}!")
            generate_item_id_from_text(
                item_text_file, item_index_file, model_gen, tokenizer
            )
    item_info = utils.ReadLineFromFile(item_index_file)
    item_map = get_dict_from_lines(item_info)

    # generate user id file
    if (
        user_id_without_target_item
    ):  # without data leakage (exclude last item from generating  user id)
        user_index_file = os.path.join(
            data_path, dataset, f"user_generative_index_phase_{phase}_no_last_item.txt"
        )
    else:  # original
        user_index_file = os.path.join(
            data_path, dataset, f"user_generative_index_phase_{phase}.txt"
        )

    if (phase == 0 and not os.path.exists(user_index_file)) or (
        phase != 0 and regenerate
    ):
        print(f"(re)generate user id with id generator phase {phase}!")
        generate_user_id_from_text(
            item_map,
            user_index_file,
            user_sequence_file,
            model_gen,
            tokenizer,
            user_id_without_target_item=user_id_without_target_item,
        )

    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)

    user_sequence_info = utils.ReadLineFromFile(user_sequence_file)
    user_sequence = get_dict_from_lines(user_sequence_info)

    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)

    if return_user_map:
        return reindex_user_sequence_dict, item_map, user_map
    else:
        return reindex_user_sequence_dict, item_map


def gram_indexing(
    data_path,
    dataset,
    model_gen,
    tokenizer,
    regenerate=True,
    phase=0,
    args=None,
    user_id_without_target_item=False,
    id_linking=False,
    compute_transition: bool = False,
    decay_tau: float = 128.0,
    decay_clip: float = 0.8,
):
    """
    Use original index to index the given user sequence dict and to represent target items.

    Return:
        - user_sequence_dict: {user_id: [item_id, item_id, ...]}
        - item2input: {item_id: item_text_input} (Used for input. 'lexical_id', 'all_text', 'wo_description')
        - item2lexid: {item_id: lexical_id} (Used for target)
    """
    user_sequence_dict, item2input, item2lexid = None, None, None

    user_sequence_file = os.path.join(data_path, dataset, "user_sequence.txt")
    user_sequence_dict = get_dict_from_lines(utils.ReadLineFromFile(user_sequence_file))
    user_sequence_dict = {k: v.split() for k, v in user_sequence_dict.items()}

    # Load lexical id for items
    if len(args.item_id_path) and os.path.exists(
        os.path.join(data_path, dataset, args.item_id_path)
    ):
        item_index_file = os.path.join(data_path, dataset, args.item_id_path)
        print(f"Load item id from {item_index_file}")
    else:
        item_index_file = os.path.join(
            data_path,
            dataset,
            f"item_generative_indexing_{args.hierarchical_id_type}.txt",
        )
        if not os.path.exists(item_index_file):
            raise FileNotFoundError(
                f"Item index file {item_index_file} does not exist. Please generate it first. or Check the path."
            )

    if (phase == 0 and not os.path.exists(item_index_file)) or (
        phase != 0 and regenerate
    ):
        print(f"> (re)generate textual id with id generator phase {phase}!")
        item_text_file = os.path.join(data_path, dataset, "item_plain_text.txt")
        generate_item_id_from_text(
            item_text_file, item_index_file, model_gen, tokenizer
        )
    item2lexid = get_dict_from_lines(utils.ReadLineFromFile(item_index_file))

    # Load item text input
    if args.item_prompt == "lexical_id":
        item2input = copy.deepcopy(item2lexid)
    elif (
        args.item_prompt
        in [
            "all_text",
            "nothing",
            "only_title",
            "only_brand",
            "only_category",
            "only_tbc",
        ]
        and args.top_k_similar_item > 0
    ):
        item2input = get_dict_with_similar_items(
            args=args,
            item2lexid=item2lexid,
            text_file="item_plain_text.txt",
            data_path=data_path,
            dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid item_prompt: {args.item_prompt}")

    if id_linking:
        for item_id, text in item2input.items():
            item2input[item_id] = f"item: {item2lexid[item_id]}; {text}"

    srciid2tgtiid2cnt = None
    if compute_transition:
        import numpy as np
        import pickle
        import pandas as pd
        from datetime import datetime
        from collections import Counter

        # Load timestamps
        df_time = pd.read_csv(os.path.join(data_path, dataset, 'timestamp.txt'), sep='\t')
        uidiid = list(zip(df_time['user_id'], df_time['item_id']))
        uidiid2datetime = dict(zip(uidiid, df_time['datetime']))

        decay_clip_str = str(decay_clip).replace('.', '')
        inter_fname = os.path.join(
            data_path, dataset,
            f'inter_dict_all_weight_{decay_tau}_{decay_clip_str}.pkl'
        )

        if os.path.exists(inter_fname):
            with open(inter_fname, 'rb') as f:
                srciid2tgtiid2cnt = pickle.load(f)
            print(f">>> Loaded inter_dict for tau={decay_tau} clip={decay_clip}")
        else:
            inter_dict = {}  # {src_iid: {tgt_iid: weighted_score_sum}}
            for uid, iids in tqdm(
                user_sequence_dict.items(),
                dynamic_ncols=True,
                desc=f'Building inter_dict tau={decay_tau} clip={decay_clip}'
            ):
                train_ids = iids[:-2]  # leave-one-out: drop valid+test tails

                # Yelp repeat-item disambiguation
                if dataset == 'Yelp':
                    tmp_counter = Counter()
                    new_iids = []
                    for iid in iids:
                        tmp_counter[(uid, iid)] += 1
                        cnt = tmp_counter[(uid, iid)]
                        new_iid = f"{iid}||{cnt}"
                        new_iids.append(new_iid)
                    train_ids = new_iids[:-2]

                for i in range(len(train_ids)):
                    for j in range(i + 1, len(train_ids)):
                        src_item = train_ids[i]
                        tgt_item = train_ids[j]

                        src_time = datetime.fromisoformat(uidiid2datetime[(uid, src_item)])
                        tgt_time = datetime.fromisoformat(uidiid2datetime[(uid, tgt_item)])
                        delta = tgt_time - src_time

                        # w(Δt) = max(exp(-Δt/τ), c)
                        score = max(np.exp(-delta.days / decay_tau), decay_clip)

                        # Strip Yelp suffix back to canonical iid for the dict key
                        if dataset == 'Yelp':
                            src_item = src_item.split('||')[0]
                            tgt_item = tgt_item.split('||')[0]

                        if src_item not in inter_dict:
                            inter_dict[src_item] = {}
                        if tgt_item not in inter_dict[src_item]:
                            inter_dict[src_item][tgt_item] = score
                        else:
                            inter_dict[src_item][tgt_item] += score

            with open(inter_fname, 'wb') as f:
                pickle.dump(inter_dict, f)
            print(f">>> Saved inter_dict to {inter_fname}")
            srciid2tgtiid2cnt = inter_dict

    return user_sequence_dict, item2input, item2lexid, srciid2tgtiid2cnt


def get_dict_from_lines(lines):
    """
    Used to get user or item map from lines loaded from txt file.
    """
    index_map = dict()
    for line in lines:
        info = line.split(" ", 1)
        index_map[info[0]] = info[1]
    return index_map


def get_dict_with_similar_items(args, item2lexid, text_file, data_path, dataset):
    top_k_similar_item = args.top_k_similar_item
    cf_model = args.cf_model

    assert (
        dataset != "Yelp" if args.item_prompt == "only_brand" else True
    ), "Yelp dataset does not have brand information."

    # read cf model's similar items
    cf_item_path = os.path.join(data_path, dataset, f"similar_item_{cf_model}.txt")
    cf_item_dict = {}
    with open(cf_item_path, "r") as file:
        for line in file:
            if line.startswith("anchor"):
                continue
            item, similar_items = line.split(" ", 1)
            similar_items = similar_items.split()
            cf_item_dict[item] = similar_items[:top_k_similar_item]

    # read item plain text
    if args.item_prompt == "all_text":
        item_text_file = os.path.join(data_path, dataset, text_file)
        item_text_dict = {}
        with open(item_text_file, "r") as file:
            for line in file:
                id_, text = line.split(" ", 1)
                item_text_dict[id_] = text.strip()
    elif args.item_prompt == "nothing":
        item_text_dict = {item_id: "" for item_id in item2lexid}
    elif args.item_prompt == "only_title":
        item_text_file = os.path.join(data_path, dataset, text_file)
        item_text_dict = {}
        attr_key = "name:" if dataset == "Yelp" else "title:"
        with open(item_text_file, "r") as file:
            for line in file:
                id_, text = line.split(" ", 1)
                text = text.split(";")
                text = [t.strip() for t in text if t.strip().startswith(attr_key)]
                item_text_dict[id_] = "; ".join(text).strip()
    elif args.item_prompt == "only_brand":
        item_text_file = os.path.join(data_path, dataset, text_file)
        item_text_dict = {}
        with open(item_text_file, "r") as file:
            for line in file:
                id_, text = line.split(" ", 1)
                text = text.split(";")
                text = [t.strip() for t in text if t.strip().startswith("brand:")]
                item_text_dict[id_] = "; ".join(text).strip()
    elif args.item_prompt == "only_category":
        item_text_file = os.path.join(data_path, dataset, text_file)
        item_text_dict = {}
        with open(item_text_file, "r") as file:
            for line in file:
                id_, text = line.split(" ", 1)
                text = text.split(";")
                text = [t.strip() for t in text if t.strip().startswith("categories:")]
                item_text_dict[id_] = "; ".join(text).strip()
    elif args.item_prompt == "only_tbc":
        item_text_file = os.path.join(data_path, dataset, text_file)
        item_text_dict = {}
        with open(item_text_file, "r") as file:
            for line in file:
                id_, text = line.split(" ", 1)
                text = text.split(";")

                if dataset in ["Beauty", "Toys", "Sports"]:
                    # find title, brand, category
                    for t in text:
                        if t.strip().startswith("title:"):
                            title = t.strip()
                        elif t.strip().startswith("brand:"):
                            brand = t.strip()
                        elif t.strip().startswith("categories:"):
                            categories = t.strip()
                    item_text_dict[id_] = f"{title}; {brand}; {categories}"
                else:
                    for t in text:
                        if t.strip().startswith("name:"):
                            title = t.strip()
                        elif t.strip().startswith("categories:"):
                            categories = t.strip()
                    item_text_dict[id_] = f"{title}; {categories}"
    else:
        raise ValueError(
            f"item_prompt should be one of ['all_text', 'nothing', 'only_title', 'only_brand', 'only_category', 'only_tbc'], but got {args.item_prompt}"
        )

    # get item2input
    item2input = {}
    for item, desc in item_text_dict.items():
        similar_items = cf_item_dict[item]
        cf_verb = [item2lexid[similar_item] for similar_item in similar_items]
        all_text = f"similar items: {', '.join(cf_verb)}; {desc}"
        item2input[item] = all_text.strip()

    return item2input


def reindex(user_sequence_dict, user_map, item_map):
    """
    reindex the given user sequence dict by given user map and item map
    """
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        uid = user_map[user]
        items = user_sequence_dict[user]
        reindex_user_sequence_dict[uid] = [item_map[i] for i in items]

    return reindex_user_sequence_dict


def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """
    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict


def generate_item_id_from_text(
    item_text_file_dir, item_id_file_dir, model_gen, tokenizer, device="cpu"
):
    """
    generate item id file from item text file
    """
    device = next(model_gen.parameters()).device
    model_gen.to("cpu")
    item_text_dict = {}
    with open(item_text_file_dir, "r") as file:
        for line in file:
            id_, text = line.split(" ", 1)
            item_text_dict[id_] = text.strip()  # Add to dictionary

    id_set = set()  # ensure no duplication
    item_id_dict = {}
    count = 0
    max_dp = 0

    for iid, text in tqdm(item_text_dict.items(), dynamic_ncols=True):
        found = False
        dp = 1.0  # penalty for diversity
        min_l = 1
        while not found:  # keep trying until generating an uniq id
            inputs = tokenizer(
                [text], max_length=256, truncation=True, return_tensors="pt"
            )
            if hasattr(model_gen, "module"):
                output = model_gen.module.generate(
                    **inputs,
                    num_beams=10,
                    num_beam_groups=10,
                    do_sample=False,
                    min_length=min_l,
                    max_length=min_l + 10,
                    diversity_penalty=dp,
                    num_return_sequences=10,
                )
            else:
                output = model_gen.generate(
                    **inputs,
                    num_beams=10,
                    num_beam_groups=10,
                    do_sample=False,
                    min_length=min_l,
                    max_length=min_l + 10,
                    diversity_penalty=dp,
                    num_return_sequences=10,
                )
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            for output in decoded_output:
                tags = re.findall(r"\b\w+\b", output)
                id = " ".join(tags)
                if id not in id_set:
                    found = True
                    id_set.add(id)
                    if dp > max_dp:
                        max_dp = dp
                    break  # if found a new id, use it
            dp += 1
            if dp >= 10:
                min_l += 10
                dp = 1.0
        item_id_dict[iid] = id

    with open(item_id_file_dir, "w") as f:
        for key, value in item_id_dict.items():
            f.write(f"{key} {value}\n")

    model_gen.to(device)
    # print('max_dp: ', max_dp)
    return True


def generate_user_id_from_text(
    item_map,
    user_index_file,
    user_sequence_file,
    model_gen,
    tokenizer,
    user_id_without_target_item=False,
):
    """
    item map: dictionary
    """

    device = next(model_gen.parameters()).device
    model_gen.to("cpu")

    user_seq_dict = (
        {}
    )  # {'A1YJEY40YUW4SE':  ['B004756YJA', 'B004ZT0SSG', 'B0020YLEYK'], ...}
    with open(user_sequence_file, "r") as file:
        for line in file:
            words = line.strip().split()
            if words:
                user_seq_dict[words[0]] = words[1:]

    for (
        user,
        items,
    ) in (
        user_seq_dict.items()
    ):  #  # {'A1YJEY40YUW4SE':  ['nail lacquer simmer and shimmer', 'red shatter crackle nail polish e55', ...
        user_seq_dict[user] = [item_map[item] for item in items]

    id_set = set()  # no duplication
    user_id_dict = {}
    id_count_dict = {}
    count = 0
    max_dp = 0

    for uid, text in tqdm(user_seq_dict.items(), dynamic_ncols=True):
        text = " ".join(text[:-1]) if user_id_without_target_item else " ".join(text)
        found = False
        dp = 1.0
        min_l = 1
        while not found:  # keep trying until generating an uniq id
            inputs = tokenizer(
                [text], max_length=256, truncation=True, return_tensors="pt"
            )
            if hasattr(model_gen, "module"):
                # Use the underlying module for generation
                output = model_gen.module.generate(
                    **inputs,
                    num_beams=10,
                    num_beam_groups=10,
                    do_sample=False,
                    min_length=min_l,
                    max_length=min_l + 10,
                    diversity_penalty=dp,
                    num_return_sequences=10,
                )
            else:
                # Model is not wrapped with DDP, use it directly
                output = model_gen.generate(
                    **inputs,
                    num_beams=10,
                    num_beam_groups=10,
                    do_sample=False,
                    min_length=min_l,
                    max_length=min_l + 10,
                    diversity_penalty=dp,
                    num_return_sequences=10,
                )
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for output in decoded_output:
                tags = re.findall(r"\b\w+\b", output)
                id = " ".join(tags)
                if id not in id_set:
                    found = True
                    id_set.add(id)
                    if dp > max_dp:
                        max_dp = dp
                    break  # if found a new id, use it
            dp += 1
            if dp >= 10:  # increase length
                min_l += 10
                dp = 1.0
        user_id_dict[uid] = id

    for key, value in user_seq_dict.items():
        if key in user_id_dict:
            user_seq_dict[key] = user_id_dict[key]
        else:
            raise ValueError(f"no user {key}")

    with open(user_index_file, "w") as f:
        for key, value in user_seq_dict.items():
            f.write(f"{key} {value}\n")
    model_gen.to(device)
    return True
