# Feasibility Review - ConvNeXt on ADNI
Author: Roan Perkins-McAuley (s48012050)

## User Need:
The primary user of this data investigation would be medical professionals involved in the diagnosis of Alzheimer’s Disease (AD), with actual patients of AD serving as indirect users. The purpose of this prototype would be to investigate the feasibility of  the ConvNeXt model in assisting medical professionals diagnosing AD, with a large focus on reducing false negatives.

## Acceptance Criteria (Scope):
- 80% test accuracy on unseen data
- low false negative rate, at least below 10%
- training time on the order of hours (less than 2 hours)
- patient level separation on datasets, no patient has data which is split between multiple datasets
- at or below 40GB peak VRAM usage during training  (limit of smaller A100 GPU's in the rangpur cluster)

## Model Choice (Hard difficulty, ConvNeXt):
Implementation will be a modified version of ConvNeXt [1] to fit the 240x256 images in the ADNI dataset. It will mirror the architectural choices of the original desing in terms of downsampling factors and channel scaling.

## Course Concepts:
Convolutions, Neural Networks & CNN's, Deep learning, Regularisation, Pattern Recognition

## Preliminary Feasibility Evidence:
- A baseline CNN trained on 1660Ti Super reached 64% test accuracy on smaller dataset
- Smoke-test run of the ConvNeXt model on the cluster reached 77% validation accuracy on full dataset. With added regularisation, a 80% test accuracy model is likely achieveable
Both of these can serve as fallbacks

## Risks:
Project scope creep and misallocation of time resources, underestimation of computation runtime, unexpected difficulties achieving the acceptance criteria

## Computing Budget:
less than 2 hours per training run on rangpur (40+ GB GPU memory), posibility of longer training on my home computer (16GB GPU memory)

## References:
[1] GeeksforGeeks (2025) ConvNeXt, GeeksforGeeks. Available at: https://www.geeksforgeeks.org/computer-vision/convnext/ (Accessed: 21 October 2026). 
