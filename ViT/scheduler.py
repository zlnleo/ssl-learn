import torch
import math


class WarmupCosineLR(
    torch.optim.lr_scheduler._LRScheduler
):

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        super().__init__(
            optimizer,
            last_epoch
        )


    def get_lr(self):

        if self.last_epoch < self.warmup_epochs:

            return [
                base_lr *
                (self.last_epoch + 1)
                /
                self.warmup_epochs

                for base_lr in self.base_lrs
            ]


        return [
            base_lr *
            0.5 *
            (
                1 +
                math.cos(
                    math.pi *
                    (
                        self.last_epoch -
                        self.warmup_epochs
                    )
                    /
                    (
                        self.max_epochs -
                        self.warmup_epochs
                    )
                )
            )

            for base_lr in self.base_lrs
        ]