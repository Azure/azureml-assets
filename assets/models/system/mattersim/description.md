MatterSim is a large-scale pretrained deep learning model for efficient materials emulations and property predictions.

MatterSim is a deep learning model for general materials design tasks. It supports efficient atomistic simulations at first-principles level and accurate prediction of broad material properties across the periodic table, spanning temperatures from 0 to 5000 K and pressures up to 1000 GPa. Out-of-the-box, the model serves as a machine learning force field, and shows remarkable capabilities not only in predicting ground-state material structures and energetics, but also in simulating their behavior under realistic temperatures and pressures. MatterSim also serves as a platform for continuous learning and customization by integrating domain-specific data. The model can be fine-tuned for atomistic simulations at a desired level of theory or for direct structure-to-property predictions with high data efficiency.

Please refer to the [MatterSim](https://arxiv.org/abs/2405.04967) manuscript for more details on the model.

- **Developed by:** Han Yang, Chenxi Hu, Yichi Zhou, Xixian Liu, Yu Shi, Jielan Li, Guanzhi Li, Zekun Chen, Shuizhou Chen, Claudio Zeni, Matthew Horton, Robert Pinsler, Andrew Fowler, Daniel Zügner, Tian Xie, Jake Smith, Lixin Sun, Qian Wang, Lingyu Kong, Chang Liu, Hongxia Hao, Ziheng Lu
- **Funded by:** Microsoft Research AI for Science
- **Model type:** Currently, we only release the models trained with **M3GNet** architecture.
- **License:** MIT License

### Model Sources

- **Repository:** <https://github.com/microsoft/mattersim>
- **Paper:** <https://arxiv.org/abs/2405.04967>

### Available Models

|                    | mattersim-v1.0.0-1M   | mattersim-v1.0.0-5M     |
| ------------------ | --------------------- | ----------------------- |
| Training Data Size | 3M                    | 6M                      |
| Model Parameters   | 880K                  | 4.5M                    |
