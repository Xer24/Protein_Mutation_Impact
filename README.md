# Protein Mutation Impact Prediction

A machine learning pipeline for predicting the functional impact of single amino acid mutations in **Green Fluorescent Protein (GFP)** using embeddings from the ESM2 protein language model.

---

## 📋 Overview

This project predicts whether a single amino acid mutation in GFP is:

- **Deleterious** (loss of function), or  
- **Tolerated** (maintains function)

The pipeline combines protein language model embeddings with classical machine learning techniques to enable structure-free mutation impact prediction.

Key components include:
- **ESM2 embeddings** to capture sequence and structural context
- **Delta embeddings** (mutant − wild-type) as mutation-sensitive features
- **Random Forest classifier** with SMOTE to address class imbalance
- **Probability calibration** for reliable confidence estimates

---

## 🎯 Project Goals

- Predict mutation effects without requiring protein structure data
- Handle severe class imbalance in experimental mutation datasets
- Provide interpretable predictions with calibrated probabilities
- Generate visualizations that characterize the mutation landscape

---

## 📊 Dataset

- **Source:** ProteinGym DMS dataset (Sarkisyan et al., 2016)  
- **Protein:** Green Fluorescent Protein (GFP)  
- **Samples:** 1,084 single amino acid substitutions  
- **Class distribution:**  
  - Tolerated: 91 (8.4%)  
  - Deleterious: 993 (91.6%)  
- **Sequence length:** 238 amino acids  

---

## 🛠️ Tech Stack

### Languages & Libraries

- Python 3.9+
- PyTorch (ESM2 model)
- scikit-learn (machine learning pipeline)
- imbalanced-learn (SMOTE)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)

### Models

- **Protein Language Model:**  
  - ESM2-t12-35M  
  - 480-dimensional embeddings  

- **Classifier:**  
  - Random Forest (200 trees, `max_depth = 15`)  

- **Calibration:**  
  - Sigmoid calibration (Platt scaling)

---

## 📈 Outputs

The pipeline produces:
- Binary mutation impact predictions
- Calibrated probability scores
- Performance metrics and evaluation plots
- Visualizations of mutation effects across the protein sequence

---

## 🔬 Notes

- The system is modular and extensible to other proteins or DMS datasets
- No structural data is required
- The framework can be adapted to alternative classifiers or embedding models
