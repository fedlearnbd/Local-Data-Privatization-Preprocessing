# Local Data Privatization Preprocessing (LDPP) Experiments

This repository contains a collection of scripts and experiments for **Local Data Privatization Preprocessing (LDPP)** in the context of federated learning. The experiments include:

- Distance analysis between original and privatized datasets.
- Federated Learning algorithms (FedAvg, FedKrum).
- LDPP integrated with Federated Learning (Ldpp-Avg, Ldpp-Krum).
- Privacy attacks such as Membership Inference Attack (MIA) and Label Flipping Attack (LFA).

The `main.py` script provides an **interactive menu** to run all experiments easily.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Menu Options](#menu-options)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [License](#license)
- [Notes](#notes)

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/fedlearnbd/Local-Data-Privatization-Preprocessing.git
cd Local-Data-Privatization-Preprocessing
2. (Optional) Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

> Make sure you have `prompt_toolkit` installed, as it is used for the interactive menu.

---

## Quick Start

1. **Prepare datasets**:

   * Place your MedNIST datasets (`pathmnist.npz`, `bloodmnist.npz`, `organamnist.npz`) in the `data/` folder.
   * Download datasets in external storage link (OneDrive). See data/Readme.md

2. **Launch the interactive menu**:

```bash
python main.py
```

3. **Run any experiment**:

   * Select the experiment by number from the menu.
   * Modify the pre-filled command if needed.
   * Press Enter to execute.

4. **Exit the menu**:

   * Choose `0` to quit.

💡 **Tip**: All experiments support parameters like dataset name, number of rounds, noise type, and epsilon for differential privacy. You can tweak them directly from the interactive prompt before running.

---

## Menu Options

| Option | Description                                                                               |
| ------ | ----------------------------------------------------------------------------------------- |
| 1      | Analysis of distances between original and private data (LDPP) - Distance Calculation     |
| 2      | Analysis of distances between original and private data (Naive DP) - Distance Calculation |
| 3      | Federated Averaging (FedAvg)                                                              |
| 4      | Federated with Krum Aggregation (FedKrum)                                                 |
| 5      | LDPP with Federated Averaging (Ldpp-Avg)                                                  |
| 6      | LDPP with Krum Aggregation (Ldpp-Krum)                                                    |
| 7      | Membership Inference Attack for FedAvg (MIA-Fedavg)                                       |
| 8      | Membership Inference Attack for Ldpp-Avg (MIA-Ldpp-Avg)                                   |
| 9      | Membership Inference Attack for Ldpp-Krum (MIA-Ldpp-Krum)                                 |
| 10     | Label Flipping Attack for FedAvg (LFA-FedAvg)                                             |
| 11     | Label Flipping Attack for Ldpp-Avg (LFA-Ldpp-Avg)                                         |
| 12     | Label Flipping Attack for Ldpp-Krum (LFA-Ldpp-Krum)                                       |

---

## Datasets

This repository uses the `pathmnist` dataset (and other MNIST variants).

> **Important:** Large datasets are not included in this repository. You may need to download them separately and place them in the `data/` folder. Example datasets:

* `bloodmnist.npz`
* `organamnist.npz`
* `pathmnist.npz`

You can store them via **Git LFS** or an external link (e.g., OneDrive, Google Drive) if needed.

---

## Requirements

* Python 3.8+
* prompt\_toolkit
* TensorFlow / PyTorch (depending on the experiments)
* NumPy, scikit-learn, and other ML libraries (see `requirements.txt`)

---

## Notes

* Each experiment is run via scripts in the `experiments/` folder.
* You can modify any experiment command directly from the interactive menu.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




