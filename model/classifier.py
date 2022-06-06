from utils.helper import *
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer, AlbertModel
from utils.metrics import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class classifier(nn.Module):
    def __init__(self, input_size, args):
        super(classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(input_size, 1)
        self.gelu = nn.GELU()

        if args.embed_mode == "albert":
            self.tokenizer = AlbertTokenizer.from_pretrained("albert-xxlarge-v2")
            self.bert = AlbertModel.from_pretrained("albert-xxlarge-v2")
        elif args.embed_mode == "bert_cased":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            self.bert = AutoModel.from_pretrained("bert-base-cased")
        elif args.embed_mode == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "allenai/scibert_scivocab_uncased"
            )
            self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        elif args.embed_mode == "biolinkbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "michiyasunaga/BioLinkBERT-large"
            )
            self.bert = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
        elif args.embed_mode == "sapbert":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )
            self.bert = AutoModel.from_pretrained(
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            )

    def forward(self, x):

        x = self.tokenizer(
            x, return_tensors="pt", padding="longest", is_split_into_words=True
        ).to(device)
        x = self.bert(**x)[0]

        x = x.transpose(0, 1)
        x = x[0, :, :]
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x[:, 0]
