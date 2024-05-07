
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from log_p_gnn.get_model import get_model

FONTSIZE = 22

plt.rcParams.update({'font.size': FONTSIZE})
plt.rcParams.update({'axes.labelsize': FONTSIZE})
plt.rcParams.update({'xtick.labelsize': FONTSIZE})
plt.rcParams.update({'ytick.labelsize': FONTSIZE})
plt.rcParams.update({'legend.fontsize': FONTSIZE})


def get_best_checkpoint_path(checkpoint_dir):
    ckpt_paths = list(Path(checkpoint_dir).glob('*.ckpt'))

    # pick the path with the smallest validation RMSE:
    val_rmses = [float(str(p).split('val_rmse=')[-1].replace('.ckpt', '')) if 'val_rmse' in str(p) else float('inf') for p in ckpt_paths]
    best_ckpt = ckpt_paths[val_rmses.index(min(val_rmses))]

    return best_ckpt

def load_from_checkpoint(checkpoint_dir):

    best_ckpt = get_best_checkpoint_path(checkpoint_dir=checkpoint_dir)

    # load the model:
    config = yaml.load((Path(checkpoint_dir).parent / 'files/log_p_gnn_config.yaml').read_text(), Loader=yaml.FullLoader)
    model = get_model(config)

    # load the weights:
    state_dict = torch.load(best_ckpt)['state_dict']

    # remove the trainling 'model.' from the keys:
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.eval()
    return model

def eval_plots(model, train_dl, val_dl, test_dl, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    # plot train, val and test scatter plots:

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    model = model.eval()
    model = model.to(device)

    for i, (dl, title) in enumerate(zip([train_dl, val_dl, test_dl], ['Train', 'Validation', 'Test'])):
        predictions, targets = [], []
        for i,g in enumerate(dl):
            with torch.no_grad():
                g = g.to(device)
                x = model(g)[:,0].detach().cpu().numpy()
                y = g.nodes['global'].data['logp'].detach().cpu().numpy()
            predictions.append(x)
            targets.append(y)


        x = np.concatenate(predictions)
        y = np.concatenate(targets)
        rmse = np.sqrt(np.mean((x - y)**2))

        ax[0][i].scatter(y, x, s=1)
        min_val = min(min(y), min(x))
        max_val = max(max(y), max(x))
        ax[0][i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        ax[0][i].set_title(title)
        ax[0][i].set_xlabel('LogP')
        ax[0][i].set_ylabel('Prediction')
        ax[0][i].text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=ax[0][i].transAxes)

        # seaborn density plot:
        sns.kdeplot(y=x, x=y, ax=ax[1][i], cmap='viridis', fill=True, thresh=0.1, levels=20, cbar=False)

        # set the same limits as above:
        ax[1][i].set_xlim(min_val, max_val)
        ax[1][i].set_ylim(min_val, max_val)
        ax[1][i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        ax[1][i].set_xlabel('Ground truth')
        ax[1][i].set_ylabel('Prediction') if i == 0 else None

    return fig, ax