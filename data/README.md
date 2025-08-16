

````markdown
## 📂 Dataset

This project uses subsets of the **MedMNIST** dataset collection.  
Due to their large size, the datasets are **not included** directly in this repository.  

### 🔹 Option 1: Automatic Download
You can run the provided script to automatically fetch the datasets into the `data/` directory:

```bash
python download_data.py
````

This will download the following files:

* `bloodmnist.npz`
* `organamnist.npz`
* `pathmnist.npz`

### 🔹 Option 2: Manual Download

Datasets can also be downloaded manually from **Google Drive** or the official **MedMNIST website**:

* [BloodMNIST (Google Drive)](https://drive.google.com/file/d/1XMw4kR_8oEQ5z7yuxkK-SMAIoJY812u3/view?usp=sharing)
* [OrganAMNIST (Google Drive)](https://drive.google.com/file/d/1aCbYfGVMLp9DK3_8mCO94PB3eoxvTpxc/view?usp=sharing)
* [PathMNIST (Google Drive)](https://drive.google.com/file/d/1hkznwSVSdOaY53SryQbp5qtt5JXAOOfR/view?usp=sharing)

Once downloaded, place the `.npz` files inside the `data/` directory:

```
project/
│── data/
│   ├── bloodmnist.npz
│   ├── organamnist.npz
│   └── pathmnist.npz
```

---

## 📖 Citation

If you use **MedMNIST**, please cite the following paper:

> Yang, J., Shi, R., Wei, D., Li, F., Wang, Z., Yu, J., … & Zhang, Y. (2023). **MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.** *Scientific Data, 10, 41.*
> [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)

BibTeX format:

```bibtex
@article{yang2023medmnist,
  title={MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jun and Shi, Rui and Wei, Donglai and Li, Fengze and Wang, Ziyang and Yu, Jiancheng and Zhang, Yue},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={41},
  year={2023},
  publisher={Nature Publishing Group}
}
```

```
```
