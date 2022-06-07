from utils.helper import *
from dataloader.dataloader import load_data
import random
import torch
import torch.nn as nn
from model.classifier import classifier
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import logging
import argparse
import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate(test_batch, args, test_or_dev):
    steps, test_loss = 0, 0
    truth = torch.empty(0).to(device)
    predicted = torch.empty(0).to(device)

    with torch.no_grad():
        for data in test_batch:
            steps += 1

            text = data[0]
            labels = torch.from_numpy(np.asarray(data[1])).to(device)

            pred = model(text)
            loss = bce(pred.float(), labels.float())
            test_loss += loss.item()

            truth = torch.cat((truth, labels), dim=0)
            predicted = torch.cat((predicted, pred), dim=0)

        predicted_labels = torch.tensor([1 if p > 0.5 else 0 for p in predicted]).to(
            device
        )
        tp = sum(predicted_labels * truth)
        fp = sum(predicted_labels * (1 - truth))
        fn = sum((1 - predicted_labels) * truth)
        tn = sum((1 - predicted_labels) * (1 - truth))
        p = tp / (tp + fp + 0.00000001)
        r = tp / (tp + fn + 0.00000001)
        f = 2 * p * r / (p + r + 0.00000001)
        result = {}
        result["p"] = p
        result["r"] = r
        result["f"] = f

        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info(
            "result: p={:.4f}, r={:.4f}, f={:.4f}".format(
                result["p"], result["r"], result["f"]
            )
        )

    return result, test_loss / steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epoch", default=10, type=int, help="number of training epoch"
    )

    parser.add_argument(
        "--batch_size",
        default=20,
        type=int,
        help="number of samples in one training batch",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=10,
        type=int,
        help="number of samples in one testing batch",
    )

    parser.add_argument(
        "--embed_mode",
        default=None,
        type=str,
        required=True,
        help="BERT or ALBERT pretrained embedding",
    )

    parser.add_argument("--lr", default=None, type=float, help="initial learning rate")

    parser.add_argument(
        "--weight_decay", default=0, type=float, help="weight decaying rate"
    )

    parser.add_argument("--seed", default=0, type=int, help="random seed initiation")

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout rate for input word embedding",
    )

    parser.add_argument(
        "--steps", default=50, type=int, help="show result for every 50 steps"
    )

    parser.add_argument(
        "--output_file",
        default="test",
        type=str,
        required=True,
        help="name of result file",
    )

    parser.add_argument(
        "--clip",
        default=0.25,
        type=float,
        help="grad norm clipping to avoid gradient explosion",
    )

    parser.add_argument(
        "--max_seq_len", default=128, type=int, help="maximum length of sequence"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = "save/" + args.output_file
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.addHandler(
        logging.FileHandler(output_dir + "/" + args.output_file + ".log", "w")
    )
    logger.info(sys.argv)
    logger.info(args)

    saved_file = save_results(
        output_dir + "/" + args.output_file + ".txt",
        header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_f \t test_f",
    )
    model_file = args.output_file + ".pt"

    train_batch, dev_batch, test_batch = load_data(args)

    logger.info("------Training------")
    if args.embed_mode == "albert":
        input_size = 4096
    elif args.embed_mode == "biolinkbert":
        input_size = 1024
    else:
        input_size = 768
    model = classifier(input_size, args)
    model.to(device)
    bce = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_result = 0
    best = None
    for epoch in range(args.epoch):
        steps, train_loss = 0, 0
        model.train()
        for data in tqdm(train_batch):

            steps += 1
            optimizer.zero_grad()
            text = data[0]
            labels = torch.from_numpy(np.asarray(data[1])).to(device)

            pred = model(text)

            loss = bce(pred.float(), labels.float())
            loss.backward()

            train_loss += loss.item()
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=args.clip
                )
            optimizer.step()

            if steps % args.steps == 0:
                logger.info(
                    "Epoch: {}, step: {} / {}, loss = {:.4f}".format(
                        epoch, steps, len(train_batch), train_loss / steps
                    )
                )

        logger.info("------ Training Set Results ------")
        logger.info("loss : {:.4f}".format(train_loss / steps))
        model.eval()
        logger.info("------ Testing ------")
        dev_f, dev_loss = evaluate(dev_batch, args, "dev")
        test_f, test_loss = evaluate(test_batch, args, "test")
        average_f1 = dev_f["f"]
        if epoch == 0 or average_f1 > best_result:
            best_result = average_f1
            best = test_f
            torch.save(model.state_dict(), output_dir + "/" + model_file)
            logger.info("Best result on dev saved!!!")
        saved_file.save(
            "{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                epoch, train_loss / steps, dev_loss, test_loss, dev_f["f"], test_f["f"]
            )
        )
    saved_file.save(
        "best test result p: {:.4f} \t r: {:.4f} \t f: {:.4f} ".format(
            best["p"], best["r"], best["f"]
        )
    )
