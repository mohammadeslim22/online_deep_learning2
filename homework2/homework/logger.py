from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Your code here - finish logging the dummy loss and accuracy

    For training, log the training loss every iteration and the average accuracy every epoch
    Call the loss 'train_loss' and accuracy 'train_accuracy'

    For validation, log only the average accuracy every epoch
    Call the accuracy 'val_accuracy'

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        my_metrics = {"train_acc": [], "val_acc": []}

        # example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            temp_train_loss = 0.9 ** (epoch + iteration / 20.0)
            temp_train_accuracy = epoch / 10.0 + torch.randn(10)

            # log the training loss
            logger.add_scalar("train_loss", temp_train_loss, global_step)
            # save the additional metrics to be averaged
            my_metrics["train_acc"].append(temp_train_accuracy.mean().item())

            #increment the global step!
            global_step += 1

        # log the average training accuracy
        avg_train_accuracy = sum(my_metrics["train_acc"]) / len(my_metrics["train_acc"])
        logger.add_scalar("train_accuracy", avg_train_accuracy, global_step)

        # example validation loop
        torch.manual_seed(epoch)
        for t in range(10):
            temp_validation_accuracy = epoch / 10.0 + torch.randn(10)

            # save the additional metrics to be averaged
            my_metrics["val_acc"].append(temp_validation_accuracy.mean().item())

        # log the average validation accuracy
        avg_val_accuracy = sum(my_metrics["val_acc"]) / len(my_metrics["val_acc"])
        logger.add_scalar("val_accuracy", avg_val_accuracy, global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
