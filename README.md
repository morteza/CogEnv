# CogEnv: Cognitive Tests Environments

The CogEnv package provides standard cognitive tests as RL environments. It uses AndroidEnv to wrap [Behaverse Cognitive Assessment Battery](https://behaverse.org).

## Quick Start

1. Install Android Studio
2. Install Android SDK
3. Create an Android Virtual Device (AVD)
4. Prepare the CogEnv environment as follows:

```bash
mamba env create -f environment.yml
# or to update existing one, use `mamba env update -f environment.yml`
```

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{Ansarinia2022CogEnv,
  title={CogEnv: A Reinforcement Learning Environment for Cognitive Tests},
  author={Morteza Ansarinia and Brice Clocher and Aurélien Defossez and Emmanuel Schmück and Pedro Cardoso-Leite},
  journal={2022 Conference on Cognitive Computational Neuroscience},
  city={San Francisco, CA},
  doi={10.32470/CCN.2022.1198-0},
  url={https://doi.org/10.32470/CCN.2022.1198-0},
  year={2022}
}
```

