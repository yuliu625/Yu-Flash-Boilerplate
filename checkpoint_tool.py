import torch

from pathlib import Path
import time


def save_checkpoint(path_to_save, model, optimizer, epoch, loss, ):
    file_path = Path(path_to_save)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        # 'scheduler': scheduler.state_dict(),
        'timestamp': time.time(),
    }, file_path)


def load_checkpoint(path_to_checkpoint, model, optimizer, ):
    checkpoint = torch.load(path_to_checkpoint)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    # if scheduler is not None:
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    return epoch, model, optimizer, loss,  # scheduler


if __name__ == '__main__':
    pass
