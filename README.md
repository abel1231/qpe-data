# Towards LLM4QPE: Unsupervised Pretraining of Quantum Property Estimation and A Benchmark

This is the official repository of the dataset for the paper [https://openreview.net/forum?id=vrBVFXwAmi](https://openreview.net/forum?id=vrBVFXwAmi).

### Quick Start
The generated dataset can be download in the [link](https://drive.google.com/file/d/1G-PB1dofJBbeyJoLPbI8lNYTrSANZpJR/view?usp=sharing).

To unzip the `.zip` file, run the code:
```bash
unzip dataset.zip
```

The folder includes:
```
dataset
├── Rydberg (dataset of Rydberg Atom model)
│   ├── pretrain (dataset used for pretraining)
│   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   ├── conditions.csv (values of physical conditions for each sample)
│   │   │   └── x.csv (measurement strings for the x-th sample)
│   ├── finetune (dataset used for finetuning)
│   │   ├── train (dataaset used for training)
│   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   ├── x.csv
│   │   │   │   │   │   └── labels (gound truth labels)
│   │   ├── test (dataaset used for evaluation)
│   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   ├── x.csv
│   │   │   │   │   │   └── labels (gound truth labels)
├── Heisenberg (dataset of anisotropic Heisenberg model)
│   ├── pretrain (dataset used for pretraining)
│   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   ├── conditions.csv (values of physical conditions for each sample)
│   │   │   ├── x.csv (measurement strings for the x-th sample)
│   ├── finetune (dataset used for finetuning)
│   │   ├── correlation (dataset for correlation function prediction)
│   │   │   ├── train (dataaset used for training)
│   │   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   │   ├── x.csv
│   │   │   │   │   │   │   └── labels (gound truth labels)
│   │   │   ├── test (dataaset used for evaluation)
│   │   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   │   ├── x.csv
│   │   │   │   │   │   │   └── labels (gound truth labels)
│   │   ├── entropy (dataset for entanglement entropy prediction)
│   │   │   ├── train (dataaset used for training)
│   │   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   │   ├── x.csv
│   │   │   │   │   │   │   └── labels (gound truth labels)
│   │   │   ├── test (dataaset used for evaluation)
│   │   │   │   ├── qxx (dataset of the system with xx qubits)
│   │   │   │   │   ├── mxx (dataset with xx random measurements)
│   │   │   │   │   │   ├── nxx (dataset with xx samples)
│   │   │   │   │   │   │   ├── conditions.csv
│   │   │   │   │   │   │   ├── x.csv
└── └── └── └── └── └── └── └── labels (gound truth labels)
```

### How to generate your own dataset
A demo used to generate small-size dataset of the anisotropic Heisenberg model is provided in `generate_heisenberg.py`. Users can adjust parameters such as the Hamiltonian, the number of qubits, the number of measurements, and the size of samples to obtain a customized dataset.
