# Optimisation of Knowledge Tracing Methods in Simulation

This repository contains the code and experimental results for the Bachelor's thesis “Optimisation of Knowledge Tracing Methods in Simulation”.

The project investigates adaptive task selection in intelligent tutoring systems using different Knowledge Tracing (KT) models. In particular, it evaluates whether selecting tasks according to their expected learning gain can reduce the number of steps required for a simulated student to reach mastery.

The experiments compare several KT models, including Bayesian Knowledge Tracing (BKT), Performance Factors Analysis (PFA), Deep Knowledge Tracing (DKT), DKT+, and TorchBKT.

---

## Repository Structure

```text
.
├── Generate_Simulated_Data.ipynb
│   └── Notebook for generating the simulated student interaction data
│
├── Models/
│   ├── BKT_pytorch.py
│   ├── ELO.py
│   ├── BKT.ipynb
│   ├── BKT_torch.ipynb
│   ├── DKT.ipynb
│   ├── DKTplus.ipynb
│   ├── PFA.ipynb
│   └── sample.ipynb
│
├── Optimization_Script/
│   └── Highest_Learning_Gain/
│       ├── Optimization_BKT.ipynb
│       ├── Optimization_Kappa_DKT.ipynb
│       ├── Optimization_Kappa_DKTplus.ipynb
│       ├── Optimization_PFA.ipynb
│       ├── Optimization_TorchBKT.ipynb
│       └── Results/
│
├── Results/
│   ├── bkt_results_scenario_*.csv
│   ├── dkt_results_scenario_*.csv
│   ├── dktplus_results_scenario_*.csv
│   ├── pfa_results_scenario_*.csv
│   ├── torchbkt_results_scenario_*.csv
│   ├── compile_results.ipynb
│   └── kt_results_summary.csv
│
├── Simulated_Data/
│   └── Training and test data for all eight scenarios
│
├── Trained_Models/
│   └── Saved trained KT models
│
├── elo_variable.json
├── task_list.txt
├── requirements.txt
└── README.md
```

---

## Models

The following Knowledge Tracing approaches are implemented and evaluated:

* BKT – Bayesian Knowledge Tracing
* TorchBKT – PyTorch-based implementation of BKT
* PFA – Performance Factors Analysis
* DKT – Deep Knowledge Tracing
* DKT+ – DKT with additional regularisation terms to reduce prediction waviness

The `Models/` directory contains the implementations and notebooks used to train and evaluate the individual models.

`ELO.py` contains the Elo-based student model used for generating simulated interaction data.

---

## Simulated Data

The experiments are conducted on simulated student interaction data.

The data generation process is implemented in:

```text
Generate_Simulated_Data.ipynb
```

The generated data is organised into eight different learning scenarios:

```text
Simulated_Data/
├── Scenario 1
├── Scenario 2
├── ...
└── Scenario 8
```

Each scenario represents a different combination of skills, tasks, task difficulties, and task-skill relationships.

The simulated interaction data contains information such as:

* student ID
* task
* skill ID
* success/failure
* interaction step

Training and test data are generated separately for the experiments.

---

## Task Selection Optimisation

The main optimisation experiments are located in:

```text
Optimization_Script/Highest_Learning_Gain/
```

The experiments investigate task selection based on the expected learning gain of a task.

The optimisation experiments are provided separately for each KT model.

---

## Experimental Scenarios

The experiments use eight simulated learning scenarios.

The scenarios differ in:

* number of skills
* number of tasks
* task difficulty
* skill-task relationships
* whether tasks cover a single skill or multiple skills

In particular, some scenarios focus on single-skill learning, while others contain multi-skill tasks.

This allows the optimisation approach to be evaluated under different learning conditions.

---

## Results

The results of the experiments are stored in:

```text
Results/
```

For each KT model, separate CSV files contain the results for the eight scenarios:

```text
bkt_results_scenario_1.csv
...
bkt_results_scenario_8.csv

dkt_results_scenario_1.csv
...
dkt_results_scenario_8.csv

dktplus_results_scenario_1.csv
...
dktplus_results_scenario_8.csv

pfa_results_scenario_1.csv
...
pfa_results_scenario_8.csv

torchbkt_results_scenario_1.csv
...
torchbkt_results_scenario_8.csv
```

The results can be combined and summarised using:

```text
Results/compile_results.ipynb
```

The compiled results are available in:

```text
Results/kt_results_summary.csv
Results/kt_results_summary.txt
```

The primary evaluation metric is the number of steps required for a student to reach mastery in all relevant skills.

---

## DKT and DKT+ Optimisation Results

Additional results for the optimisation experiments are stored in:

```text
Optimization_Script/Highest_Learning_Gain/Results/
```

This directory contains separate result files for DKT and DKT+ as well as combined result files.

For example:

```text
dkt_results_combined.txt
dktplus_results_combined.txt
```

The corresponding scenario-specific CSV files contain the detailed experimental results.

---

## Requirements

The required Python packages are listed in:

```text
requirements.txt
```

The project was developed and tested with Python 3.12.

To install the dependencies, create a virtual environment and run:

```bash
pip install -r requirements.txt
```

For example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
venv\Scripts\activate
```

---

## Reproducing the Experiments

The general workflow is:

### 1. Generate the simulated data

Run:

```text
Generate_Simulated_Data.ipynb
```

This generates the simulated student interaction data for the different scenarios.

### 2. Train the KT models

The model notebooks in:

```text
Models/
```

can be used to train and evaluate the individual KT models.

Trained models are stored in:

```text
Trained_Models/
```

### 3. Run the optimisation experiments

The optimisation notebooks are located in:

```text
Optimization_Script/Highest_Learning_Gain/
```

Run the notebook corresponding to the desired KT model.

### 4. Compile the results

After the experiments have been completed, use:

```text
Results/compile_results.ipynb
```

to combine the scenario-specific results and generate the summary tables.

---

---

## Notes

The repository contains both source notebooks and generated experimental results. The CSV files in `Results/` represent the results used for the analysis in the thesis.

Some files, such as `__pycache__/` and `.pyc` files, are automatically generated by Python and are not required to reproduce the experiments.

For a clean clone of the repository, these files can be excluded using a `.gitignore` file.
