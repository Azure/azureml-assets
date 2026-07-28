### Testing Data, Factors & Metrics

#### Testing Data

To evaluate the model performance, we created the following test sets

- **MPtrj-random-1k:** 1k structures randomly sampled from MPtrj dataset
- **MPtrj-highest-stress-1k:** 1k structures with highest stress magnitude sampled from MPtrj dataset
- **Alexandria-1k:** 1k structures randomly sampled from Alexandria
- **MPF-Alkali-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)
- **MPF-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)
- **Random-TP:** For detailed description of the generation of the dataset, please refer to the SI of the [MatterSim manuscript](https://arxiv.org/abs/2405.04967)

We released the test datasets in pickle files and each of them contains the `ase.Atoms` objects. To access the structures and corresponding labels in the datasets, you do use the following snippet to get started,

```python
import pickle
from ase.units import GPa

atoms_list = pickle.load(open("/path/to/datasets.pkl", "rb"))
atoms = atoms_list[0]

print(f"Energy: {atoms.get_potential_energy()} eV")
print(f"Forces: {atoms.get_forces()} eV/A")
print(f"Stress: {atoms.get_stress(voigt=False)} eV/A^3, or {atoms.get_stress(voigt=False)/GPa}")
```

#### Metrics

We evaluate the performance by computing the mean absolute errors (MAEs) of energy (E), forces (F) and stress (S) of each structures within the same dataset. The MAEs are defined as follows,
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_E=\frac{1}{N}\sum_{i}^N\frac{1}{N_{at}^{(i)}}|E_i-\tilde{E}_i|" alt="MAE_E equation">
</p>
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_F=\frac{1}{N}\sum_i^N\frac{1}{N_{at}^{(i)}}\sum_{j}^{N^{(i)}_{at}}||F_{ij}-\tilde{F}_{ij}||_2," alt="MAE_F equation">
</p>
<p align="center">
      <img src="https://latex.codecogs.com/svg.latex?\mathrm{MAE}_S=\frac{1}{N}\sum_i^{N}||S_{i}-\tilde{S}_{i}||_2," alt="MAE_S equation">
</p>
where N is the number of structures in the same dataset, <img src="https://latex.codecogs.com/svg.image?\inline&space;&space;N_{at}^{(i)}"> is the number of atoms in the i-th structure and E, F and S represent ground-truth energy, forces and stress, respectively.

### Results

| Dataset              | Dataset Size | MAE               | mattersim-v1.0.0-1M | mattersim-v1.0.0-5M |
| -------------------- | ------------ | ----------------- | ------------ | ------------ |
| MPtrj-random-1k      | 1000         | Energy [eV/atom]  | 0.030        | 0.024        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.149        | 0.109        |
|                      |              | Stress [GPa]      | 0.241        | 0.186        |
| MPtrj-high-stress-1k | 1000         | Energy [eV/atom]  | 0.110        | 0.108        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.417        | 0.361        |
|                      |              | Stress [GPa]      | 6.230        | 6.003        |
| Alexandria-1k        | 1000         | Energy [eV/atom]  | 0.058        | 0.016        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.086        | 0.042        |
|                      |              | Stress [GPa]      | 0.761        | 0.205        |
| MPF-Alkali-TP        | 460          | Energy [eV/atom]  | 0.024        | 0.021        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.331        | 0.293        |
|                      |              | Stress [GPa]      | 0.845        | 0.714        |
| MPF-TP               | 1069         | Energy [eV/atom]  | 0.029        | 0.026        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.418        | 0.364        |
|                      |              | Stress [GPa]      | 1.159        | 1.144        |
| Random-TP            | 693          | Energy [eV/atom]  | 0.208        | 0.199        |
|                      |              | Forces [eV/<img src="https://latex.codecogs.com/svg.latex?\AA" alt="\AA">] | 0.933        | 0.824        |
|                      |              | Stress [GPa]      | 2.065        | 1.999        |
