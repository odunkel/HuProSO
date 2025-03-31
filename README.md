# HuProSO3: Normalizing Flows on the Product Space of SO(3) Manifolds for Probabilistic Human Pose Modeling

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2404.05675)

[Olaf Dünkel](https://odunkel.github.io/), [Tim Salzmann](https://tim-salzmann.github.io/), [Florian Pfaff](https://de.linkedin.com/in/florian-pfaff-02a16a66).

This repository contains the codebase of the [CVPR24 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Dunkel_Normalizing_Flows_on_the_Product_Space_of_SO3_Manifolds_for_CVPR_2024_paper.pdf) "Normalizing Flows on the Product Space of SO(3) Manifolds for Probabilistic Human Pose Modeling", where we introduce HuProSO3, a normalizing flow model that operates on a high-dimensional product space of SO(3) manifolds, modeling the distribution of human joint rotations.

This repository includes code for training and evaluation of the pre-trained models for unconditional prior and pose estimation from 2D or 3D keypoints.



## Usage

### Installation Instructions

In your prefered environment, install all required packages using `pip install -r requirements.txt`.
Then, install the `hp` library via ```pip install .```

We use [Body Visualizer](https://github.com/nghorbani/body_visualizer/) for rendering of humans. Follow their instructions for installations if this functionality is desired.

### Preprocessing of Data
- Download the AMASS data from the [official homepage](https://amass.is.tue.mpg.de/).
- Extract the data using `bash scripts/extract_amass_datasets.sh`.
- Preprocess the data using `bash scripts/preprocess_amass_data.sh`.


### Training
The unconditional prior can be trained using `python scripts/train_prior.py`.
Training of the model inverse kinematics, i.e. 3D keypoints to SMPL joint rotations, can be performed via `python scripts/train_SO3_ik.py`.
For training with 2D keypoint condition, assign `conditioning.conditioning_modality='2D'` in `config/config.yaml`. For randomly masking the condition during training, set `conditioning.mask=true`.



### Inference and Results
Model inference and example evaluations are illustrated in `explore/evaluate.ipynb`.


## Citation
If you find our work useful, please cite our paper:
```bibtex
@inproceedings{dunkel2024normalizing,
  title={Normalizing flows on the product space of SO(3) manifolds for probabilistic human pose modeling},
  author={D{\"u}nkel, Olaf and Salzmann, Tim and Pfaff, Florian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2285--2294},
  year={2024}
}
```

## Acknowledgements

Our code uses components of the following open-source projects: [RotationNormFlow](https://github.com/PKU-EPIC/RotationNormFlow), [Adversarial Parametric Pose Prior](https://github.com/cvlab-epfl/adv_param_pose_prior/tree/main), [Body Visualizer](https://github.com/nghorbani/body_visualizer/), [SIMPLify](https://github.com/githubcrj/simplify), and [ImplicitPDF](https://github.com/google-research/google-research/tree/master/implicit_pdf). We thank the developers of these resources.
