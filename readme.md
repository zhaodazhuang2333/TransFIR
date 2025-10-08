# TransFIR

This repository contains the code implementation for the paper '*Inductive Reasoning on Temporal Knowledge Graphs with Emerging Entities*', which explores the inductive reasoning for emerging entities without historical interactinos in TKGs.


<div align="center">
  <img src="TransFIR.png" alt="Logo" width="50%" />
</div>


## Project Structure
```
├── data/               # Directory for storing datasets
├── results/            # Directory for experiment results
├── README.md           # Project documentation
├── models.py           # Model implementation
├── utils.py            # Utility functions
└── main.py             # Training script
```

## Experiment Environment

- python 3.9

- pytorch 2.1+

- torch-geometric 2.4.0


## Train and Test

### data preprocess

First, get the interaction chain for each dataset

```
python data_process.py  --dataset {dataset}  --T 14
```

Then, get the textual embedding for each dataset, 
```
python word_embedding.py  --dataset {dataset}  --bert_model_path {your_bert_path}
```

### train and test
```
python main.py --dataset {dataset} --max_length {max_length} --num_layers {num_laters} --hidden_dim {hidden_dim} --num_code {num_code} 
```
For each dataset, you can run:

```
python main.py --dataset ICEWS14 --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 50


python main.py --dataset ICEWS18 --max_length 15 --num_layers 2 --hidden_dim 768 --num_code 50


python main.py --dataset ICEWS05-15 --max_length 10 --num_layers 3 --hidden_dim 1024 --num_code 30


python main.py --dataset GDELT --max_length 30 --num_layers 3 --hidden_dim 768 --num_code 100
```
