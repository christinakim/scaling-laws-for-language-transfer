# Scaling Laws for Language Transfer Learning
Code and models from the blog post [Scaling Laws for Language Transfer Learning](https://christina.kim/2021/04/11/scaling-laws-for-language-transfer-learning/)

## Motivation
Building upon work from [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) (Hernandez et. al. 2021), my experiments focused on exploring the relationships between fine-tuning on non-English languages. My experiments try to answer the question: How much does pre-training on English help when transferring across different languages as we vary the dataset size and model size?

## Usage
This repo contains the code for: 
1) Reproducing pre-trained decoder-only transformers using hyperparameters from [Scaling Laws for Neural Languages](https://arxiv.org/abs/2001.08361) but trained on [OpenWebtext2](https://openwebtext2.readthedocs.io/en/latest/) instead of WebText 
2) Reproducing language transfer experiments for pre-trained English models to Chinese, Spanish, and German texts 

All English pre-trained models were trained for 26 billion tokens with no repeats: 
- [x6small](https://huggingface.co/christina/decoder-only-transformer-x6small) 3.3M non-embedding parameters
- [x5small](https://huggingface.co/christina/decoder-only-transformer-x5small) 16M non-embedding parameters
- [x4small](https://huggingface.co/christina/decoder-only-transformer-x4small) 39M non-embedding parameters
- [x3small](https://huggingface.co/christina/decoder-only-transformer-x3small) 51M non-embedding parameters
- [x2small](https://huggingface.co/christina/decoder-only-transformer-x2small) 70M non-embedding parameters
- [small](https://huggingface.co/christina/decoder-only-transformer-small) 124M non-embedding parameters
