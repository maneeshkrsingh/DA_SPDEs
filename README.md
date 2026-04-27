# DA-SPDEs: Forward Models for Data Assimilation in Stochastic PDEs

## 👋 Overview
This repository provides forward models for stochastic partial differential equations (SPDEs) used in data assimilation frameworks.

It supports:
- Nudging-based data assimilation  
- Particle filtering methods  
- Ensemble-based simulation workflows  

---

## 🔬 Motivation
Many real-world systems (e.g., fluid dynamics, climate models) are naturally described by stochastic PDEs. Data assimilation combines these models with observational data to estimate system states under uncertainty.

This repository focuses on building reliable forward solvers that can be integrated into data assimilation pipelines.

---

## ⚙️ Key Features
- Modular forward models for SPDE systems  
- Designed for integration with data assimilation methods  
- Supports ensemble simulations  
- Compatible with scalable solvers (Firedrake / PETSc)  

---

## 📁 Repository Structure

    DA_SPDEs/
    ├── bs_assimilate_errorcompare.py   # Example assimilation/analysis script
    ├── other scripts and models        # Forward models and experiments

---

## 🧪 Methods
The implementations combine:
- Stochastic PDE modelling  
- Numerical discretisation (FEM / DG concepts)  
- Ensemble-based simulation  

These forward models are designed to work with:
- Nudging schemes  
- Particle filters  
- Bayesian inference methods  

---

## 🚀 Getting Started

### Requirements
- Python  
- NumPy / SciPy  
- Firedrake (or similar FEM framework)  
- PETSc  

### Example usage

    python bs_assimilate_errorcompare.py

---

## 📊 Applications
- Data assimilation for stochastic systems  
- Uncertainty quantification  
- Scientific machine learning  
- Fluid and geophysical modelling  

---

## 👤 Author
Maneesh Kumar Singh  

- https://github.com/maneeshkrsingh  
- https://maneeshkrsingh.github.io  

---

## 📄 License
MIT License
