from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, num_epochs, power, min_lr=0, last_epoch=-1, verbose=False):
        self.num_epochs = num_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [max(self.min_lr, base_lr * ((1 - self.last_epoch / self.num_epochs) ** self.power)) for base_lr in self.base_lrs]
