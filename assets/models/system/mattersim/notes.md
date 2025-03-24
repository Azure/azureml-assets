## Resources

* [An example that runs the model on Foundry](https://github.com/microsoft/mattersim/blob/main/notebooks/foundry-mattersim-phonon.ipynb).
* [Implementation of the MatterSim model](https://github.com/microsoft/mattersim)
* [Documentation of the MatterSim implementation](https://microsoft.github.io/mattersim/)
* [Paper with detailed evaluation](https://arxiv.org/abs/2405.04967)

## Intended Uses

The MatterSim model is intended for property predictions of materials.

### Direct Use

The model is used for materials simulation and property prediciton tasks. An interface to atomic simulation environment is provided. Examples of direct usages include but not limited to

- Direct prediction of energy, forces and stress of a given materials
- Phonon prediction using finite difference
- Molecular dynamics

### Out-of-Scope Use

The model only supports atomistic simulations of materials and molecules. Any attempt and interpretation beyond that should be avoided.
The model does not support generation of new materials as it is designed for materials simulation and property prediction only.
The model is intended for research and experimental purposes. Further testing/development are needed before considering its application in real-world scenarios.

## Contact Model Provider

- Han Yang (<hanyang@microsoft.com>)
- Ziheng Lu (<zihenglu@microsoft.com>)

## Technical Specifications

### Model Architecture and Objective

The checkpoints released in this repository are those trained on an internal implementation of the **M3GNet** architecture.

#### Software

- Python == 3.9

## Citation

**BibTeX:**

```
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```

## Bias, Risks, and Limitations

The current model has relatively low accuracy for organic polymeric systems.
Accuracy is inferior to the best (more computationally expensive) methods available.
The model is trained on a specific variant of Density Functional Theory (PBE) that has known limitations across chemical space which will affect accuracy of prediction, such as the ability to simulate highly-correlated systems. (The model can be fine-tuned with higher accuracy data.)
The model does not support all capabilities of some of the latest models such as predicting Born effective charges or simulating a material in an applied electric field.
We have evaluated the model on many examples, but there are many examples that are beyond our available resources to test.

### Recommendations

For any appications related simulations of surfaces, interfaces, and systems with long-range interactions, the results are often qualitatively correct. For quantitative results, the model needs to be fine-tuned.
