# Generative Iris Prior Embedded Transformer for Iris Restoration

[ICME 2023] This is an implementation of [Generative Iris Prior Embedded Transformer for Iris Restoration](https://arxiv.org/abs/2407.00261)

- Overview of Gformer architecture. Gformer consists of Transformer encoder and generative iris prior embedded decoder. They are bridged by the iris feature modulator.
<p align="center">
<img src="/asset/Gformer.png">
</p>

- Qualitative comparison.
<p align="center">
<img src="/asset/Comparison.png">
</p>

- ROC curve for comparison.
<p align="center">
<img src="/asset/roc.png">
</p>

## Requirements

 - See `requirements.txt`

### Installation

```bash
pip install basicsr
pip install -r requirements.txt
```

## Instructions
 - Global hyperparameters are configured in files under `options/`
 - See README.md files under `experiments/`, there are some pretrained models to download.

## Examples

```bash
python src/train.py -opt options/train_gformer.yml --auto_resume
```
## BibTeX

    @InProceedings{huang2023gformer,
        author = {Yubo Huang and Jia Wang and Peipei Li and Liuyu Xiang and Peigang Li and Zhaofeng He},
        title = {Generative Iris Prior Embedded Transformer for Iris Restoration},
        booktitle={The IEEE Conference on Multimedia and Expo (ICME)},
        year = {2023}
    }

## Acknowledgement
Yubo Huang, Jia Wang, Peipei Li, Liuyu Xiang, Peigang Li, Zhaofeng He
