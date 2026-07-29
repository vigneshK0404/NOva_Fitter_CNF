# NOvA Conditional Normalizing Flow Seeder

This repository implements a simulation-based inference workflow designed to reduce the phase-space search required by the NOvA global sterile-neutrino fitter.

## Overview

The central question is whether data generated under the three-flavor hypothesis, including systematic and statistical uncertainties, can be better described by a point in the four-flavor sterile-neutrino parameter space.

The global sterile fitter was designed to answer this question, but a direct search over the full parameter space is computationally expensive. This project trains a Conditional Normalizing Flow (CNF) to learn the relationship

```text
data -> generating parameters
```

After training, the CNF can process a new experiment and return a ranked list of candidate seed points. These seeds are then passed to the fitter so that it can converge more quickly than it would from the null hypothesis or from a broad unguided search.

The full workflow spans two systems:

```text
NOvA environment
    generate pseudoexperiments
        |
        | scp
        v
Heimdall
    preprocess data
    train CNF
    generate candidate seeds
        |
        | scp
        v
NOvA environment
    run fitter
    evaluate inference quality
```

## Requirements

### NOvA environment

Data generation and fitter validation require:

- CAFAna
- NuXAna
- PISCES

Refer to the NOvA software setup guide for instructions on installing and configuring these dependencies.

### Heimdall

The CNF side requires:

- Python
- Conda
- PyTorch
- nflows
- NumPy
- SciPy
- scikit-learn
- uproot

A Conda environment file is included in the repository.

---

# 1. Data Generation

Pseudoexperiments are generated using:

```text
macros/generate_Preds.C
```

Run the macro with an appropriate spacing parameter:

```bash
cafe -bq --nuxana macros/generate_Preds.C <spacing>
```

The macro contains two important settings:

- `unique`: Number of unique parameter points to generate.
- `repeat`: Number of statistically fluctuated datasets generated from each unique parameter point.

For each unique parameter point, the script generates the corresponding expected data and then produces `repeat` Poisson-fluctuated realizations.

Conceptually:

```text
one unique theta point
    -> one expected spectrum
    -> multiple Poisson-fluctuated experiments
```

The current evidence suggests that exposing the CNF to more data generally helps. However, no large improvement in inference quality was observed when increasing from approximately 30 repeats to 150 repeats. This should be investigated more carefully before drawing a firm conclusion.

## Grid Submission

The generation macro can be submitted to the grid with:

```bash
submit_cafana.py \
    --nuxana \
    --onsite_only \
    --lifetime 72000 \
    --memory 1900 \
    --njobs 100 \
    --rel development \
    --snapshot \
    --outdir /pnfs/nova/scratch/users/karthiva/NOvA_CNF \
    --user_tarball CNFGen.tar.bz2 \
    macros/generate_Preds.C <spacing>
```

Adjust the job count, output directory, resource requests, tarball, and spacing parameter as needed.

---

# 2. Heimdall Setup

Move to your personal RAID workspace:

```bash
cd /raid/$USER
```

Create the Conda environment using the provided environment file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cnf
```

Create a directory named `scripts` and copy the contents of `heimdall_scripts` into it:

```bash
mkdir -p scripts
cp -r heimdall_scripts/* scripts/
```

From the top-level workspace, run:

```bash
python3 scripts/consts.py
```

This creates the working directories expected by the pipeline.

The repository assumes that large generated datasets are stored under:

```text
/share/lazy/$USER/data
```

rather than `/raid/$USER`, because of storage limitations on the RAID filesystem.

The code, model checkpoints, plots, and other smaller outputs remain under the user's RAID workspace.

---

# 3. Transfer Generated Data

Copy the generated ROOT files from the NOvA environment into:

```text
/share/lazy/$USER/tmp_root_files
```

For example:

```bash
scp <nova-host>:<path-to-root-files>/*.root \
    <heimdall>:/share/lazy/$USER/tmp_root_files/
```

Replace the host and path placeholders with the appropriate NOvA machine and output location.

---

# 4. Preprocessing and Training

Run the Heimdall scripts in the following order.

## 4.1 Clean and package ROOT files

```bash
python3 scripts/cleanROOT.py
```

`cleanROOT.py` reads the generated ROOT files, cleans the data, and packages them into `.npz` files.

After confirming that the `.npz` files were created successfully, the original ROOT files may be deleted if storage is a concern.

## 4.2 Calculate statistics and standardize data

```bash
python3 scripts/readCNFROOT.py
```

`readCNFROOT.py`:

- calculates the means and standard deviations of the data and parameter distributions,
- stores the standardization statistics,
- standardizes the data and parameter values,
- splits the dataset into chunks that are easier to load into memory.

Training, validation, and inference must all use the same saved standardization statistics.

## 4.3 Train the CNF

```bash
python3 scripts/CNFtrainer.py <run_validation> <model_name>
```

The script expects two command-line arguments:

```text
<run_validation>  Boolean controlling whether inference validation is run after training.
<model_name>      Name used for the model output directory.
```

For training only:

```bash
python3 scripts/CNFtrainer.py False my_model
```

Model checkpoints, validation plots, logs, and a PDF representation of the model architecture are written to:

```text
/raid/$USER/Models/<model_name>
```

---

# 5. Model Validation

Validation has two parts:

1. Inspect the validation plots produced during training.
2. Test whether the generated CNF seeds improve the sterile-neutrino fit.

## 5.1 Generate validation experiments

Return to the NOvA environment and use:

```text
fitter.C
```

This macro contains two primary functions:

- `create_exp`
- `check_inference`

First run `create_exp` to generate the experiment or experiments used for inference testing.

Copy the resulting ROOT file to Heimdall under:

```text
inference_data/
```

For example:

```bash
scp <nova-host>:<path-to-generated-inference-file>.root \
    <heimall>/raid/$USER/inference_data/
```

## 5.2 Generate CNF representatives

On Heimdall, run:

```bash
python3 scripts/validateCNF.py
```

This produces a ROOT file containing representative parameter points selected from the CNF posterior samples.

For each experiment, the CNF:

1. samples candidate parameter points,
2. identifies posterior modes,
3. selects a high-likelihood representative from each mode,
4. ranks the representatives by CNF log probability.

Copy the resulting ROOT file back to the NOvA environment and place it inside the expected `data/` directory.

## 5.3 Check fitter performance

Run `check_inference` from `fitter.C`.

This procedure evaluates whether the candidate points proposed by the CNF produce better fit results than the null-hypothesis seed.

The primary comparison is:

```text
null-hypothesis seed
        vs.
CNF-proposed candidate seeds
```

The current implementation is successful when the CNF provides fitter seeds that lead to better minima than the null hypothesis.

---

# 6. Faster Validation Across Multiple Models

When testing many models against the same inference dataset, the same generated experiments can be reused.

Set the first `CNFtrainer.py` argument to `True`:

```bash
python3 scripts/CNFtrainer.py True <model_name>
```

This trains the model and automatically runs the CNF validation workflow using the existing inference data.

This avoids manually rerunning `validateCNF.py` after every new training run.

---

# 7. Current Results

After extensive testing, the CNF generally provides fitter seeds that perform better than the null hypothesis.

However, the seeds do not yet consistently reach the best four-flavor point found by the global sterile fitter. The global fitter can still identify lower-chi-square solutions than those proposed by the CNF.

The CNF is therefore useful as a phase-space reduction and fitter-initialization method, but it should not currently be interpreted as a replacement for the full global sterile fit.

---

# 8. Known Issues and Areas for Improvement

## Phase-space coverage

Poor phase-space coverage has already been investigated and does not appear to explain the observed performance gap.

The current generation procedure samples each of the six parameters independently in one dimension. If more uniform multidimensional coverage is desired, this could be replaced with a Sobol sequence or another low-discrepancy sampling method.

## Generating point versus best-fit point

A major possibility is that the parameter point used to generate an experiment is not necessarily the same point that minimizes the fitter objective for that statistically fluctuated experiment.

In other words:

```text
generating parameter point != fitted best-fit parameter point
```

This may be especially important when systematic effects, degeneracies, and statistical fluctuations are present.

The CNF is currently trained using generating parameters as truth labels. If ,however, these truth labels do not correspond to the best fit points. It could explain the issue.

One way to investigate this is to:

1. generate many pseudoexperiments,
2. run the global sterile fitter on each experiment,
3. record the fitted global minimum,
4. compare the generating point, CNF representatives, and fitter minimum.

Initial work toward this study is contained in:

```text
diagnose.C
```

in the NOvA environment.

The relevant bash script for running these global fits on the grid is also provided as 

```text
generate.sh
```
It may also be useful to generate data without statistical fluctuations and test whether the fitted best-fit point then agrees with the generating point.

## Model limitations

If the mismatch is caused by the model rather than the labels, possible explanations include:

- insufficient model capacity,
- inadequate hyperparameter optimization,
- data leakage,
- preprocessing inconsistencies,
- standardization errors,
- an unsuitable posterior clustering or representative-selection method,
- a mismatch between the CNF training objective and fitter performance.

These possibilities have not yet been conclusively separated.

## Repeat count

The effect of the number of statistical repeats per unique parameter point remains uncertain.

Experiments comparing approximately 30 and 150 repeats did not show a large difference in inference quality, but this result should be confirmed using a controlled comparison with identical model architecture, training budget, and validation experiments.

---

# 9. Repository Structure

The exact structure may evolve, but the main components are:

```text
macros/
    generate_Preds.C      Generate pseudoexperiments in the NOvA environment.

heimdall_scripts/
    consts.py             Define paths, constants, and create the workspace.
    cleanROOT.py          Clean and package ROOT files into NPZ files.
    readCNFROOT.py        Calculate statistics, standardize, and split data.
    CNFtrainer.py         Train and optionally validate the CNF.
    validateCNF.py        Generate ranked representative fitter seeds.

fitter.C                  Create validation experiments and evaluate CNF seeds.
diagnose.C                Diagnostic studies of generating and best-fit points.
environment.yml           Conda environment specification.
generate.sh               Submits global fits to the grid
```

---

# 10. Important Assumptions

This repository was built for use within the NOvA research environment and is not intended to be a fully portable standalone package.

The workflow assumes:

- access to the NOvA software environment,
- access to Fermilab computing resources,
- access to Heimdall,
- access to `/raid/$USER`,
- access to `/share/lazy/$USER`,
- manual file transfer between the NOvA and Heimdall environments,
- compatible versions of the software listed in the provided Conda environment.

Paths and commands may need to be updated if the group computing environment changes.

---

# Authors

Repository created by:

- Vigneshwar Karthikeyan
- Dr. Adam J. Aurisano
