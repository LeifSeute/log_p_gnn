## Installation

### Requirements

- dgl
- hydra
- pytorch
- pysmiles
- cgsmiles

### Installation

#### CPU MODE (for inference)
```bash
pip install -r installation/cpu_requirements.txt
pip install -e .
```

#### GPU MODE (for training)
```bash
pip install -r installation/requirements.txt
pip install -e .
```

See `installation/README.md`.



### Inference

See `scripts/inference.py` for an example of how to use the model for infering logP values from cgsmiles directly.


### Training

```bash
python experiments/train.py --config-name combined
```

### Evaluation

```bash
python experiments/evaluate.py evaluate.ckpt_path=exported_ckpts/combined/best.ckpt
```