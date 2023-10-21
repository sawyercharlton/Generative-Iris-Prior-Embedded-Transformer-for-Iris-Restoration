# Generative Iris Prior Embedded Transformer for Iris Restoration

[ICME 2023] This is an implementation of [Generative Iris Prior Embedded Transformer for Iris Restoration](https://sawyercharlton.github.io/home/files/Generative_Iris_Prior_Embedded_Transformer_for_Iris_Restoration.pdf)

- Overview of Gformer architecture. Gformer consists of Transformer encoder and generative iris prior embedded decoder. They are bridged by the iris feature modulator.
<p align="center">
<img src="/asset/gformer.pdf">
</p>

- Qualitative comparison.
<p align="center">
<img src="/asset/comparison.pdf">
</p>

- ROC curve for comparison.
<p align="center">
<img src="/asset/roc.pdf">
</p>

## Requirements

See `requirements.txt`

### Installation

1. Install dependent packages

    ```bash
    pip install basicsr
    pip install -r requirements.txt
    ```

## Instructions
 - Global hyperparameters are configured in `options/`

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

## :e-mail: Contact

If you have any question, please email `yubo.huang@hotmail.com`.