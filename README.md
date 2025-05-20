# TrialBench: AI-Ready Clinical Trial Datasets



This repository contains code for training and testing benchmark models on Trialbench datasets. TrialBench comprises 23 AI-ready clinical trial datasets for 8 well-defined tasks: clinical trial duration forecasting, patient dropout rate prediction, serious adverse event, all-cause mortality rate prediction, trial approval outcome prediction, trial failure reason identification, eligibility criteria design, and drug dose finding. The provided scripts facilitate the evaluation of various machine learning algorithms, enabling researchers to assess their performance on different clinical trial phases and tasks. 



## 🚀 Installation 
We recommend creating a dedicated virtual environment (such as conda) with Python 3.7+ to ensure consistent performance. Once your environment is ready, install the required dependencies:
```
pip install -r requirements.txt
```

## 🔩 Download
All necessary supporting documents can be downloaded from this [link](https://drive.google.com/drive/folders/1fp350IUj284EnTHVgSWtq9qIq0Mlbjg9?usp=sharing). Place them into the `data/` folder.

## 📚 Trialbench
For quick exploration, toy samples are available in [Trialbench](https://github.com/ML2Health/ML2ClinicalTrials/tree/main/Trialbench). You can also [Download All Data](https://zenodo.org/records/15426202/files/all_data.zip?download=1) at once.

#### Sub-Task Datasets

1. [Trial Duration Forecasting](https://zenodo.org/records/15426202/files/trial-duration-forecasting.zip?download=1)
2. [Patient Dropout Event Forecasting](https://zenodo.org/records/15426202/files/patient-dropout-event-forecasting.zip?download=1)
3. [Serious Adverse Event Forecasting](https://zenodo.org/records/15426202/files/serious-adverse-event-forecasting.zip?download=1)
4. [Mortality Event Prediction](https://zenodo.org/records/15426202/files/mortality-event-prediction.zip?download=1)
5. [Trial Approval Forecasting](https://zenodo.org/records/15426202/files/trial-approval-forecasting.zip?download=1)
6. [Trial Failure Reason Identification](https://zenodo.org/records/15426202/files/trial-failure-reason-identification.zip?download=1)
7. [Eligibility Criteria Design](https://zenodo.org/records/15426202/files/eligibility-criteria-design.zip?download=1)
8. [Drug Dose Finding](https://zenodo.org/records/15426202/files/drug-dose-prediction.zip?download=1)

Here’s a refined and expanded Usage section with a clear explanation of base_name and how to use it to run different experiments:

## 💻 Usage

To run an experiment for mortality rate prediction, navigate to the `AI4Trial` directory and execute:

```bash
cd AI4Trial
python learn_multi_model.py --base_name mortality_rate --phase 'Phase 1' --exp Temp
```
#### Configurable Parameters
--base_name: Specifies the task to run (see the available options below).

--phase: Defines the experimental phase (e.g., 'Phase 1').

--exp: Sets the output dir for tracking.


#### Available base_name Options

The base_name parameter determines which dataset and task to use. Below are the supported tasks:

| base_name                  | Task Description                                      |
|----------------------------|-------------------------------------------------------|
| mortality_rate             | Mortality event prediction                            |
| serious_adverse_rate       | Serious adverse event forecasting                     |
| patient_dropout_rate       | Patient dropout event forecasting                     |
| duration                   | Trial duration forecasting                            |
| outcome                    | Trial outcome prediction                              |
| failure_reason             | Trial failure reason identification                   |
| serious_adverse_rate_yn    | Binary classification for serious adverse events      |
| patient_dropout_rate_yn    | Binary classification for patient dropout events      |
| mortality_rate_yn          | Binary classification for mortality prediction        |
| dose                       | Drug dose finding (regression)                        |
| dose_cls                   | Drug dose finding (classification )                   |

To run experiments for other tasks, replace mortality_rate in the command with the corresponding base_name from the table above. For example, to run a serious adverse event forecasting experiment:

```
python learn_multi_model.py --base_name serious_adverse_rate --phase 'Phase 2' --exp TestRun
```





