from train import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import logging


@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="evaluate")
def main(cfg: DictConfig) -> None:

    ckpt_path = Path(cfg.evaluate.ckpt_path)

    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    
    if (ckpt_path.parent/'config.yaml').exists():
        ckpt_cfg = OmegaConf.load(ckpt_path.parent/'config.yaml')
    else:
        raise NotImplementedError(f"Checkpoint config not found at {ckpt_path.parent/'config.yaml'}")

    # replace the checkpoint dir in ckpt_cfg with the one from the checkpoint path:
    ckpt_cfg.experiment.checkpointer.dirpath = ckpt_path.parent

    # determine the accelerator:
    ckpt_cfg.experiment.trainer.accelerator = cfg.evaluate.accelerator
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU.")
        ckpt_cfg.experiment.trainer.accelerator = 'cpu'

    # set data.extend_train_set to 1:
    ckpt_cfg.data.extend_train_epoch = 1

    # init the experiment:
    exp = Experiment(cfg=ckpt_cfg)

    # test the model
    exp.test(ckpt_path=ckpt_path, eval_all=True)


if __name__ == "__main__":
    main()