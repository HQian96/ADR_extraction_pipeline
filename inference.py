import json
import argparse
from model.pfn import *
from model.classifier import *
import torch
from transformers import AlbertTokenizer, AutoTokenizer
import re
from tqdm import tqdm


def map_origin_word_to_bert(words, tokenizer):
    bep_dict = {}
    current_idx = 1

    for word_idx, word in enumerate(words):
        bert_word = tokenizer.tokenize(word)
        word_len = len(bert_word)
        bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
        current_idx = current_idx + word_len

    return bep_dict


def inference_PFN_2(args, sentences: list, threshold=0.5):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.embed_mode == "albert":
        input_size = 4096
    elif args.embed_mode == "biolinkbert":
        input_size = 1024
    else:
        input_size = 768
    with open("data/ADE/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open("data/ADE/rel2idx.json", "r") as f:
        rel2idx = json.load(f)
    idx2ner = {v: k for k, v in ner2idx.items()}
    idx2rel = {v: k for k, v in rel2idx.items()}
    model = PFNn(args, input_size, ner2idx, rel2idx)
    model_file = "save/" + args.model_file + "/" + args.model_file + ".pt"
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()
    if args.embed_mode == "albert":
        tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
    elif args.embed_mode == "bert_cased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    elif args.embed_mode == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    elif args.embed_mode == "biolinkbert":
        tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    elif args.embed_mode == "sapbert":
        tokenizer = AutoTokenizer.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )

    output = []
    with torch.no_grad():
        for sentence in tqdm(sentences):
            ents = []
            rels, rels2 = [], []
            target_sent = re.findall(r"\w+|[^\w\s]", sentence)
            sent_bert_ids = tokenizer(
                target_sent, return_tensors="pt", is_split_into_words=True
            )["input_ids"].tolist()
            sent_bert_ids = sent_bert_ids[0]
            sent_bert_str = []
            for i in sent_bert_ids:
                sent_bert_str.append(tokenizer.convert_ids_to_tokens(i))
            bert_len = len(sent_bert_str)

            mask = torch.ones(bert_len, 1).to(device)
            ner_score, re_score, re_score2 = model(target_sent, mask)

            ner_score = torch.where(
                ner_score >= threshold,
                torch.ones_like(ner_score),
                torch.zeros_like(ner_score),
            )
            re_score = torch.where(
                re_score >= threshold,
                torch.ones_like(re_score),
                torch.zeros_like(re_score),
            )
            re_score2 = torch.where(
                re_score2 >= threshold,
                torch.ones_like(re_score2),
                torch.zeros_like(re_score2),
            )

            entity = (ner_score == 1).nonzero(as_tuple=False).tolist()
            relation = (re_score == 1).nonzero(as_tuple=False).tolist()
            relation2 = (re_score2 == 1).nonzero(as_tuple=False).tolist()

            word_to_bep = map_origin_word_to_bert(target_sent, tokenizer)
            bep_to_word = {word_to_bep[i][0]: i for i in word_to_bep.keys()}

            entity_names, entity_names2 = {}, {}
            for en in entity:
                type = idx2ner[en[3]]
                start = None
                end = None
                if en[0] in bep_to_word.keys():
                    start = bep_to_word[en[0]]
                if en[1] in bep_to_word.keys():
                    end = bep_to_word[en[1]]
                if start == None or end == None:
                    continue

                entity_str = " ".join(target_sent[start : end + 1])
                entity_names[entity_str] = start
                entity_names2[entity_str] = end
                ents.append((entity_str, type))

            for rel in relation:
                type = idx2rel[rel[3]]

                e1 = None
                e2 = None

                if rel[0] in bep_to_word.keys():
                    e1 = bep_to_word[rel[0]]
                if rel[1] in bep_to_word.keys():
                    e2 = bep_to_word[rel[1]]
                if e1 == None or e2 == None:
                    continue

                subj = None
                obj = None

                for en, start_index in entity_names.items():
                    if en.startswith(target_sent[e1]) and start_index == e1:
                        subj = en
                    if en.startswith(target_sent[e2]) and start_index == e2:
                        obj = en

                if subj == None or obj == None:
                    continue
                rels.append((subj, type, obj))

            for rel in relation2:
                type = idx2rel[rel[3]]
                e1 = None
                e2 = None
                if rel[0] in bep_to_word.keys():
                    e1 = bep_to_word[rel[0]]
                if rel[1] in bep_to_word.keys():
                    e2 = bep_to_word[rel[1]]
                if e1 == None or e2 == None:
                    continue

                subj = None
                obj = None
                for en, start_index in entity_names2.items():
                    if (
                        en.split(" ")[-1].startswith(target_sent[e1])
                        and start_index == e1
                    ):
                        subj = en
                    if (
                        en.split(" ")[-1].startswith(target_sent[e2])
                        and start_index == e2
                    ):
                        obj = en

                if subj == None or obj == None:
                    continue
                rels2.append((subj, type, obj))
            rels = list(set(rels) & set(rels2))
            output.append({"text": sentence, "entities": ents, "relations": rels})
    return output


def inference_classifier(args, sentences: list, threshold=0.5):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.embed_mode == "albert":
        input_size = 4096
    elif args.embed_mode == "biolinkbert":
        input_size = 1024
    else:
        input_size = 768

    model = classifier(input_size, args)
    model_file = (
        "save/" + args.model_file + "/" + args.model_file + "classifier" + ".pt"
    )
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    output = []
    with torch.no_grad():
        for sentence in tqdm(sentences):
            target_sent = re.findall(r"\w+|[^\w\s]", sentence)
            if target_sent:
                pred = model(target_sent)
                pred = pred[0]
                if pred > threshold:
                    output.append(sentence)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, required=True, help="loading pre-trained model files"
    )

    parser.add_argument(
        "--hidden_size", type=int, default=300, help="hidden size of the model"
    )

    parser.add_argument(
        "--dropconnect", type=float, default=0.1, help="dropconnect on encoder"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout on word embedding and task units",
    )

    parser.add_argument(
        "--embed_mode",
        default=None,
        type=str,
        required=True,
        help="BERT or ALBERT pretrained embedding",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="data/pubmed.txt",
        required=True,
        help="Path to the text file for inference",
    )

    args = parser.parse_args()
    with open(args.text, "r", encoding="utf-8") as f:
        pubmed = [line.rstrip("\n") for line in f]
    results = inference_classifier(args, pubmed)
    results = inference_PFN_2(args, results)
    json.dump(results, open("data/" + args.model_file + "_results.json", "w"))
