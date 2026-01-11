# CoV-SNN
A Contrastive Learning Framework for Efficient Viral Escape Prediction

This repository contains the implementation of CoV-SNN, a framework for predicting viral escape mutations using contrastive learning on Spike protein sequences.

## Sequence of Experiments

The experiments are designed to be run in the following order:

### 1. Pretraining
Train the CoV-RoBERTa model from scratch using Masked Language Modeling (MLM) on viral sequences.

**Instructions:**
Run the following steps in order to process data and train the model:
```bash
# Step 1: Sample sequences from the raw dataset
python pretraining/step1_sampler.py

# Step 2: Create masked inputs for MLM
python pretraining/step2_masker.py

# Step 3: Train the tokenizer
python pretraining/step3_tokenizer.py

# Step 4: Run MLM pretraining
python pretraining/step4_mlm_trainer.py
```

### 2. Variant Classification
Fine-tune the pretrained CoV-RoBERTa model (and other baselines) to classify viral variants (Alpha, Beta, Delta, Gamma, Omicron).

**Instructions:**
To train the main CoV-RoBERTa classifier:
```bash
python variant_classification/covroberta.py
```
Other baselines (ESM2, ProtBERT, etc.) can be run similarly using their respective scripts in the `variant_classification` directory.

### 3. Zero-shot Variant Classification
Train CoV-SNN using contrastive learning to distinguish between variants in the embedding space and evaluate zero-shot performance.

**Instructions:**
Run the training and testing script:
```bash
python zeroshot_variant_classification/train_test_zero_shot.py
```
You can adjust parameters like loss function, margins, and pooling modes via command line arguments (see script for details).

### 4. Wet Lab Verified Escape Prediction
Evaluate the model's ability to predict viral escape using wet-lab verified datasets (e.g., Spike data from Baum et al. and Spike RBD data from Greaney et al.).

**Instructions:**
Run the training and testing script for wet-lab data:
```bash
python escape_prediction_wet_lab/train_test_wet_lab.py
```

### 5. Escape Prediction (Our Dataset)
Compute Constrained Semantic Change Search (CSCS) scores to predict escape potential on our collected dataset. This involves calculating semantic change (grammaticality vs. semantic shift).

**Instructions:**
Run the computation script with a specified checkpoint ID:
```bash
python escape_prediction/CoVSNN_compute.py 4
```

### 6. Supporting Analyses
Perform interpretability studies, confidence interval calculations, and statistical significance tests.

**Instructions:**
Run the desired analysis script:
```bash
# Layer-wise attribution analysis
python supporting_analyses/captum_layer_attr.py

# Position-wise attribution analysis
python supporting_analyses/captum_position_attr.py

# Calculate confidence intervals
python supporting_analyses/confidence_intervals.py

# Statistical significance analysis for CSCS scores
python supporting_analyses/statistical_significance_cscs.py
```

### Note

Files from the `mlm_checkpoints/`, `checkpoints/`, and `outputs/` directories are not included in the repository due to size constraints. Please contact the authors to request access.