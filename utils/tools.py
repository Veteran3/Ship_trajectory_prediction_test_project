import numpy as np
import torch
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    """
    调整学习率
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'cosine':
        lr = args.learning_rate * (0.5 * (1 + np.cos((epoch - 1) / args.train_epochs * np.pi)))
        optimizer.param_groups[0]['lr'] = lr
        print(f'Updating learning rate to {lr}')
        return
    else:
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


def visual(true, pred, name='./pic/test.pdf'):
    """
    可视化结果
    
    Args:
        true: [T, N, D] - 真实轨迹
        pred: [T, N, D] - 预测轨迹
        name: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 可视化第一艘船的经纬度
    plt.subplot(1, 2, 1)
    plt.plot(true[:, 0, 0], true[:, 0, 1], 'b-', label='GroundTruth', linewidth=2)
    plt.plot(pred[:, 0, 0], pred[:, 0, 1], 'r--', label='Prediction', linewidth=2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Ship Trajectory (Ship 0)')
    plt.legend()
    plt.grid(True)
    
    # 可视化所有船的轨迹
    plt.subplot(1, 2, 2)
    for i in range(min(5, true.shape[1])):  # 最多画5艘船
        plt.plot(true[:, i, 0], true[:, i, 1], '-', alpha=0.6, label=f'GT Ship {i}')
        plt.plot(pred[:, i, 0], pred[:, i, 1], '--', alpha=0.6, label=f'Pred Ship {i}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Multi-Ship Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    plt.close()