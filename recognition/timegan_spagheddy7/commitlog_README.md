## Development and Commit Log
## 26 October 2026: Data pipeline and initial feasibility
## Commit: 6fbd4c9
## Message: Initialise TimeGAN project with data pipeline and feasibility checks
Today, I established the structure of the project and prepared the LOBSTER AMZN data pipeline, which included checking the supplied order book data, transforming prices and volumes, constructing data partitions and preparing sequence windows.

I also added the data-checking notebook, initial visualisations and dataset tests, the feasibility review was completely earlier however committed today which contained an overview of the approach and computational considerations.

I began writing the report with the abstract and introduction.

## 31 October 2026: Autoregressive baseline
## Commit: 4f9a67f
## Message: Add autoregressive baseline and validation report
Today I implemented a stochastic vector autoregressive baseline model which included ridge regression regularisation and tuning for the lag orders and residual-based generations. I also added scripts for training and prediction, evaluation functions and tests as well as a results notebook.

The baseline was completed to provide a simpler comparison for the TimeGAN network down the line. I prepared foundational validation report for the orderbook invalidity metrics, spread and return distributions and autocorrelation diagnostics.

Implemented results in the final report.

## 31 October 2026: Feasibility and development documentation
## Commit: e254e6a
## Message: Update feasibility draft and development log
This was the 1st of second planned commits for today with this first one being small documentation and development log changes to reflect progress. I did this to keep it separate from the baseline implementation commit.

Implemented results in the final report.

## 31 October 2026: Depth similarity and comparison protocol
## Commit: 856f34d
## Message: Add depth SSIM and shared comparison setup
Today I added evaluations for the SSIM scoring including a depth-heatmap and expanded further evaluation tests. I also introduced a common baseline for comparison protocols and updated the baseline notebooks with updated figures and cached results.

This allowed for consistency in evaluation settings for subsequent comparisons which further included validation window evaluations and depth-map scaling.
Implemented results in the final report.

## 2 November 2026: TimeGAN implementation and experiments
## Commit: 2718795
## Message: Add TimeGAN experiments and reproducible validation reports
I implemented the multi network PyTorch TimeGAN system and began the experiments with ordering penalties, starting book info and removal of temporal supervision.
Further, I added comparisons across training seeds and failure diagnostics/visualisations along with results and report caching. This was so that the notebooks could consistently present saved evidence on the varying experiments.

ChatGPT assisted in test cases passing 64 project tests.

Overall, the experiments showed solid trade-offs between orderbook validity, structural and distributional similarity, and temporal behaviour discussed further in the report.
Implemented results in the final report.

## 4 November 2026 : Reconstruction studies, ladder decoder and interrupted volume experiment
## Commit: dec51d7
## Message: Record ladder studies and incomplete volume-conditioning pilot
Today, I kept the reconstruction diagnostics along with the ladder-decoder experiments and associated notebook results. The ladder encoder was important to the report to analyse how enforcing price ordering through output construction impacted the target metrics.

The initial starting volume experiment was incomplete due to the training being cutoff but training progress was retained (however it was presented with this limitation).

Implemented results in the final report.

## 4–5 November 2026: Completed work awaiting the final commit
For these days, I added full training state persistence to support experimental recovery for some final results, replaying the original experiment with preserved checkpoints. I then completed the matched starting-volume comparisons and managed to improve SSIM scores from 0.1788 to 0.2829. Unfortunately other error metrics slightly worsened so the updated model could not pass the entire improvement screen.
The notebooks were updated to distinguish this additional final comparison, keeping the other validation-selected checkpoints separate and consistent. 
I also used prior results to retain some CPU and GPU efficiency metrics including timing and memory usage and completed the final report.

## 6 November 2026 — Final submission commit
## Commit hash: 
## Message: Finalise TimeGAN recovery results and submission documentation
The final commit is made to include the completed recovery implementations with tests and validation reports. The results notebook is also updated, and the main README is updated with the final report (as I wrote it in Word for efficiency). It contains relevant results with figures and tables, the original feasibility review, instructions, references, and an AI usage disclosure statement as I used ChatGPT to assist.

## Artificial Intelligence Usage Disclosure
OpenAI’s ChatGPT was used in assistance with writing code for the python scripts and Jupyter notebooks. This assisted in the implementation and diagnostics of the research question and helped with recovering saved experiments for the purpose of analysis. It also assisted in debugging and regression tests for checkpoint audits and replays. The implementation was checked using automated tests with checkpoint audits and replay comparisons, with results being crosschecked with the efficiency of ChatGPT.
