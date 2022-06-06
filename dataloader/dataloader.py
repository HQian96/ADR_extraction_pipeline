import json
from utils.helper import *
from transformers import AlbertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random


class dataprocess(Dataset):
    def __init__(self, data, embed_mode, max_seq_len):
        self.data = data
        self.len = max_seq_len
        if embed_mode == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
        elif embed_mode == "bert_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif embed_mode == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased"
            )
        elif embed_mode == "biolinkbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "michiyasunaga/BioLinkBERT-large"
            )
        elif embed_mode == "sapbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        ner_labels = self.data[idx][1]
        rc_head_labels = self.data[idx][2]
        rc_tail_labels = self.data[idx][3]

        if len(words) > self.len:
            words, ner_labels, rc_head_labels, rc_tail_labels = self.truncate(
                self.len, words, ner_labels, rc_head_labels, rc_tail_labels
            )

        sent_str = " ".join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        # bert_len = original sentence + [CLS] and [SEP]

        word_to_bep = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform(ner_labels, word_to_bep)
        rc_head_labels = self.rc_label_transform(rc_head_labels, word_to_bep)
        rc_tail_labels = self.rc_label_transform(rc_tail_labels, word_to_bep)

        return (words, ner_labels, rc_head_labels, rc_tail_labels, bert_len)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label), 3):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i]][0] + 1
            end = word_to_bert[ner_label[i + 1]][0] + 1
            new_ner_labels += [sta, end, ner_label[i + 2]]

        return new_ner_labels

    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label), 3):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i]][0] + 1
            e2 = word_to_bert[rc_label[i + 1]][0] + 1
            new_rc_labels += [e1, e2, rc_label[i + 2]]

        return new_rc_labels

    def truncate(self, max_seq_len, words, ner_labels, rc_head_labels, rc_tail_labels):
        truncated_words = words[:max_seq_len]
        truncated_ner_labels = []
        truncated_rc_head_labels = []
        truncated_rc_tail_labels = []
        for i in range(0, len(ner_labels), 3):
            if ner_labels[i] < max_seq_len and ner_labels[i + 1] < max_seq_len:
                truncated_ner_labels += [
                    ner_labels[i],
                    ner_labels[i + 1],
                    ner_labels[i + 2],
                ]

        for i in range(0, len(rc_head_labels), 3):
            if rc_head_labels[i] < max_seq_len and rc_head_labels[i + 1] < max_seq_len:
                truncated_rc_head_labels += [
                    rc_head_labels[i],
                    rc_head_labels[i + 1],
                    rc_head_labels[i + 2],
                ]

        for i in range(0, len(rc_tail_labels), 3):
            if rc_tail_labels[i] < max_seq_len and rc_tail_labels[i + 1] < max_seq_len:
                truncated_rc_tail_labels += [
                    rc_tail_labels[i],
                    rc_tail_labels[i + 1],
                    rc_tail_labels[i + 2],
                ]

        return (
            truncated_words,
            truncated_ner_labels,
            truncated_rc_head_labels,
            truncated_rc_tail_labels,
        )


def ade_preprocess(data, dataset):
    processed = []
    for dic in data:
        text = dic["tokens"]
        ner_labels = []
        rc_head_labels = []
        rc_tail_labels = []
        entity = dic["entities"]
        relation = dic["relations"]

        for en in entity:
            ner_labels += [en["start"], en["end"] - 1, en["type"]]

        for re in relation:
            subj_idx = re["head"]
            obj_idx = re["tail"]
            subj = entity[subj_idx]
            obj = entity[obj_idx]
            rc_head_labels += [subj["start"], obj["start"], re["type"]]
            rc_tail_labels += [subj["end"] - 1, obj["end"] - 1, re["type"]]

        overlap_pattern = False
        if dataset == "ADE":
            for i in range(0, len(ner_labels), 3):
                for j in range(i + 3, len(ner_labels), 3):
                    if is_overlap(
                        [ner_labels[i], ner_labels[i + 1]],
                        [ner_labels[j], ner_labels[j + 1]],
                    ):
                        overlap_pattern = True
                        break
        if overlap_pattern == True:
            continue

        processed += [(text, ner_labels, rc_head_labels, rc_tail_labels)]
    return processed


def dataloader(args, ner2idx, rel2idx):
    path = "data/" + args.data

    raw_ade_data = json_load(path, "ade_triples.json")
    raw_n2c2_data = json_load(path, "n2c2_triples.json")
    random.shuffle(raw_ade_data)
    random.shuffle(raw_n2c2_data)
    data = raw_ade_data
    random.shuffle(data)
    split = int(0.1 * len(data))

    train_data = data[2 * split :] + raw_n2c2_data
    random.shuffle(train_data)
    test_data = data[:split]
    dev_data = data[split : 2 * split]

    train_data = ade_preprocess(train_data, args.data)
    test_data = ade_preprocess(test_data, args.data)
    dev_data = ade_preprocess(dev_data, args.data)

    train_dataset = dataprocess(train_data, args.embed_mode, args.max_seq_len)
    test_dataset = dataprocess(test_data, args.embed_mode, args.max_seq_len)
    dev_dataset = dataprocess(dev_data, args.embed_mode, args.max_seq_len)
    collate_fn = collater(ner2idx, rel2idx)

    train_batch = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_batch = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dev_batch = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_batch, test_batch, dev_batch


def dataloader_10fold(args, ner2idx, rel2idx, i):
    path = "data/" + args.data

    raw_ade_data = json_load(path, "ade_triples.json")
    raw_n2c2_data = json_load(path, "n2c2_triples.json")
    random.shuffle(raw_ade_data)
    random.shuffle(raw_n2c2_data)
    data = raw_ade_data
    random.shuffle(data)
    split = int(0.1 * len(data))
    if i == 0:
        train_data = data[split:] + raw_n2c2_data
        random.shuffle(train_data)
        dev_data = data[:split]
    elif i == 9:
        train_data = data[: 9 * split] + raw_n2c2_data
        random.shuffle(train_data)
        dev_data = data[9 * split :]
    else:
        dev_data = data[i * split : (i + 1) * split]
        train_data = data[: i * split] + data[(i + 1) * split :] + raw_n2c2_data
        random.shuffle(train_data)

    train_data = ade_preprocess(train_data, args.data)
    dev_data = ade_preprocess(dev_data, args.data)

    train_dataset = dataprocess(train_data, args.embed_mode, args.max_seq_len)
    dev_dataset = dataprocess(dev_data, args.embed_mode, args.max_seq_len)
    collate_fn = collater(ner2idx, rel2idx)

    train_batch = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dev_batch = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_batch, dev_batch


def load_data(args):
    with open("data/classification/yes.json", "r") as f:
        y = json.load(f)
    with open("data/classification/no.json", "r") as f:
        n = json.load(f)
    random.shuffle(y)
    random.shuffle(n)
    split1 = int(0.1 * len(y))
    split2 = int(0.1 * len(n))
    train_dataset = y[2 * split1 :] + n[2 * split2 :]
    test_dataset = y[:split1] + n[:split2]
    dev_dataset = y[split1 : 2 * split1] + n[split2 : 2 * split2]
    train_dataset = [
        (item["tokens"], item["label"])
        for item in train_dataset
        if len(item["tokens"]) < args.max_seq_len
    ]
    test_dataset = [
        (item["tokens"], item["label"])
        for item in test_dataset
        if len(item["tokens"]) < args.max_seq_len
    ]
    dev_dataset = [
        (item["tokens"], item["label"])
        for item in dev_dataset
        if len(item["tokens"]) < args.max_seq_len
    ]

    collate_f = collate()
    train_batch = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_f,
    )
    test_batch = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_f,
    )
    dev_batch = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_f,
    )
    return train_batch, dev_batch, test_batch
