# GRUT datasets

This directory contains preprocessed datasets used for training and evaluating the GRUT model.

## 📁 Dataset Structure

The `rec_datasets/` directory contains 4 datasets organized as follows:

```
rec_datasets/
├── Beauty/          # Amazon Beauty products dataset
├── Sports/          # Amazon Sports & Outdoors dataset
├── Toys/            # Amazon Toys & Games dataset
└── Yelp/            # Yelp business reviews dataset
```

## 📄 File Descriptions

Each dataset folder contains the following files:

### 1. `user_sequence.txt`
- **Format**: Space-separated User ASIN and Item ASINs per line
- **Description**: User interaction sequences ordered chronologically (first token is User ASIN, followed by Item ASINs)
- **Example**: `A1YJEY40YUW4SE B004756YJA B004ZT0SSG B0020YLEYK 7806397051 B002WLWX82`
  - `A1YJEY40YUW4SE`: User ASIN
  - `B004756YJA B004ZT0SSG B0020YLEYK 7806397051 B002WLWX82`: Item ASINs in chronological order
- **Usage**: Used for train, valid and test based on leave-one-out setting

### 2. `item_plain_text.txt`
- **Format**: One item per line with structured metadata
- **Description**: Concatenated rich item descriptions including title, brand, categories, description, price, sales rank, etc. (Yelp dataset has different fields).
- **Fields**:
  - `title`: Product name
  - `brand`: Manufacturer/brand name
  - `categories`: Product categories
  - `description`: Detailed product description
  - `price`: Product price
  - `salesrank`: Sales ranking in category
  ...
- **Example**:
  ```
  B00DJQQEGQ title: shea butter 100 natural african ultra rich raw shea nuts...; brand: na; categories: beauty, skin care, body, moisturizers, body butter; description: would you like to have a glowing looking skin...; price: 17.97; salesrank: beauty: 20924
  ```

### 3. `timestamp.txt`
- **Format**: Tab-separated, with header `user_id\titem_id\ttimestamp\tdatetime`
- **Description**: Raw interaction timestamps used to build user-level temporal context (`C_u`) and to compute item-level transition decay scores (paper Eq.1) at indexing time.
- **Example**:
  ```
  user_id	item_id	timestamp	datetime
  A00414041RD0BXM6WK0GX	B007IY97U0	1405296000	2014-07-14 00:00:00+00:00
  ```

### 4. `temporal_feature.tsv`
- **Format**: Tab-separated, with header `user_id\titem_id\tabsolute_time\trelative_time\ttarget_relative_time`
- **Description**: Pre-computed temporal feature strings used inside the user-level prompt `C_u` (paper §4.2). Each row gives, for one (user, item) pair, an absolute date string, a delta vs. the previous item (`relative_time`), and a delta to the held-out target (`target_relative_time`). The `--temporal_feature` argument selects which subset to render into the prompt.
- **Example**:
  ```
  user_id	item_id	absolute_time	relative_time	target_relative_time
  A00414041RD0BXM6WK0GX	B007IY97U0	2014-07-14	None	before less than an hour from now
  ```

### 5. `user_generative_index_phase_0.txt`
- **Format**: User ASIN followed by space-separated lexical tokens
- **Description**: User textual ID (`user_lex_id`) that replaces the raw User ASIN inside the prompt template (`{user_lex_id}`). Mirrors `item_generative_indexing_*.txt` but for users.
- **Example**:
  ```
  A1YJEY40YUW4SE nail lacquer simmer and shimmer red shatter
  ```

### 6. `item_generative_indexing_hierarchy_v1_*.txt`
- **Format**: Item ID and hierarchical textual tokens separated by `|`
- **Description**: Textual IDs for each item, organized in hierarchy. The example below uses the GRAM-published hierarchical ID; **GRUT works with any item-id scheme** as long as a `item_generative_indexing_*.txt` file is supplied and selected via `--hierarchical_id_type` (e.g., `idgenrec`, `elmrec`, `p5_sid_tso`, `item_bm25_l7_split`, …). 
- **Filename Pattern**: `item_generative_indexing_hierarchy_v1_c{cluster}_l{level}_len{length}_split.txt`
  - `c{cluster}`: Number of clusters (e.g., c32, c128)
  - `l{level}`: Hierarchy levels (e.g., l5, l7, l9)
  - `len{length}`: Maximum sequence length of item texts during hierarchical clustering (e.g., len128, len32768)
- **Usage**: Used for hierarchical semantics indexing.
- **Example**:
  ```
  B000P24EI2 |▁loss|▁brake|▁fur|▁consolid|ren|▁scalp|▁profound
  ```
  - `B000P24EI2`: Item ID
  - `|▁loss|▁brake|▁fur|▁consolid|ren|▁scalp|▁profound`: Hierarchical textual tokens separated by `|`

## 🗃️ Derived Caches (not tracked)

These are produced on first run and live under each dataset folder; they are listed in `.gitignore`:
- `timestamp_{train,valid,test}.tsv` — leave-one-out splits of `timestamp.txt`, built by the trend-rerank runner.
- `timestamp_day{N}_score_log_max.{npy,pkl}`, `timestamp_day{N}_score_asin.pkl`, `timestamp_day{N}_score_user_*_pkl` — per-user log-max trend score arrays (paper Eq.9), indexed by item ASIN / user id.
- `inter_dict_all_weight_{tau}_{clip}.pkl` — time-decayed transition score dictionary `src_iid → {tgt_iid → score}` built by `gram_indexing(compute_transition=True, ...)` using paper Eq.1.

## 📚 Data Sources
- **Amazon Review Data**: [Julian McAuley's Amazon Dataset](https://jmcauley.ucsd.edu/data/amazon/)
- **Yelp Open Dataset**: [Yelp Dataset Challenge](https://business.yelp.com/data/resources/open-dataset/)
