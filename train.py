import argparse
import logging
import torch.nn as nn
import torch.optim as optim
from model.model import GRU4REC
from model.loss import *
from model.metric import accuracy
from data_loader import RSC15DataLoader
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(description='pytorch implementation of GRU4REC')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('-e', '--epochs', default=32, type=int,
                    help='number of total epochs (default: 32)')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved', type=str,
                    help='directory of saved model (default: saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets/data', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.1)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')


def main(args):
    # Model
    model = GRU4REC(6741, 1000, 1)
    # model.summary()

    # A logger to store training process information
    train_logger = Logger()

    # Specifying loss function, metric(s), and optimizer
    loss = top1_loss
    metrics = [accuracy]
    optimizer = optim.Adam(model.parameters())

    # Data loader and validation split
    data_loader = RSC15DataLoader(args.data_dir, args.batch_size, shuffle=True)
    # valid_data_loader = data_loader.split_validation(args.validation_split) TODO: define validation set, loader

    # An identifier for this training session
    training_name = type(model).__name__

    # Trainer instance
    trainer = Trainer(model, loss, metrics,
                      data_loader=data_loader,
                    #   valid_data_loader=valid_data_loader,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      train_logger=train_logger,
                      save_dir=args.save_dir,
                      save_freq=args.save_freq,
                      resume=args.resume,
                      verbosity=args.verbosity,
                      training_name=training_name,
                      with_cuda=not args.no_cuda,
                      monitor='accuracy',
                      monitor_mode='max')

    # Start training!
    trainer.train()

    # See training history
    print(train_logger)


if __name__ == '__main__':
    main(parser.parse_args())
