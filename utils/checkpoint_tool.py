import torch

from pathlib import Path


def save_checkpoint(path_to_save, model, optimizer, ):
    file_path = Path(path_to_save)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
    }, file_path)


def load_checkpoint(path_to_checkpoint, model, optimizer, ):
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # if scheduler is not None:
    #     scheduler.load_state_dict(checkpoint['scheduler'])
    return  model, optimizer,  # scheduler


if __name__ == '__main__':
    pass
