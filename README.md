# Project Name: Brain Tumor Detection and Segmentation System

## 👥 Team Members
- Member 1 – Head detection
- Member 2 – CT tumor detection
- Member 3 – MRI tumor detection
- Member 4 – Tumor Segementation

---

## 📌 Project Overview
This project aims to build a machine learning model that predict Brain Tumors and Segment the tumor. We follow best practices in reproducible data science with a modular codebase, automated testing, and experiment tracking.

---
![c285d8a7-95bb-4416-819a-386bfb0617e7](https://github.com/user-attachments/assets/b53f58bd-d015-4865-87b3-70493225f4b0)

## 📁 Repository Structure

```plaintext
DSGP-Brain-Tumor_Detection
├── configs
├── data
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs
├── experiments
├── models
├── notebooks
│   ├── eda/
│   ├── prototyping/
│   └── reports
├── results
├── scripts
├── src
│   ├── data/
│   │   └── organize.py
│   └── utils/
│       └── utils_config.py
├── tests
│   └── test_organize.py
├── .gitignore
├── .python-version
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── setup.cfg
└── uv.lock
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

- git clone https://github.com/Chanura04/DSGP-Brain-Tumor-Detection.git
- cd DSGP-Brain-Tumor-Detection

### 2. Install dependencies

- poetry export -f requirements.txt --output requirements.txt
- pip install -r requirements.txt

### 3. Run the pipeline

- make all

---

## 📄 License
MIT License
