# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

## HipMRI segmentation submission

Open [the project README](HipMRI_UNet_s4906923/README.md) for the technical report, results and running instructions. The same report is available as [a PDF](HipMRI_UNet_s4906923/REPORT.pdf).

The submitted code runs the Standard 3D U-Net and Improved v3. Their Validation predictions have been checked against the original experiment results. Saved results from all ten model runs remain in the project's `evidence/` directory for comparison.

### Project structure

Main files and folders:

```text
HipMRI_UNet_s4906923/
├── README.md          # Full report, results and running instructions
├── REPORT.pdf         # PDF export of the project README
├── modules.py         # Standard 3D and Improved v3 model components
├── dataset.py         # Patient splits, paired preprocessing and patch sampling
├── train.py           # Training, recovery, evaluation and saved-record inspection
├── predict.py         # Fixed-weight inference, overlays and label-volume export
├── utils.py           # Loss functions, metrics and inference helpers
├── config/            # Fixed patient split and training settings
├── evidence/          # Ten-run result records and source/checkpoint identities
├── figures/           # Model diagram, learning curves and real MRI comparisons
├── tests/             # Synthetic geometry, model, metric and recovery checks
├── requirements.txt   # Pinned core Python dependencies
└── export_report.py   # Rebuild REPORT.pdf from README.md and its figures
```
