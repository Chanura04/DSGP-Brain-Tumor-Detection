# SOURCE CODE

### Purpose:
Core package of the project. Contains all reusable, production-ready Python code.

### Common Files:
- src/data/
  - data_loading.py : Functions to load datasets.
  - preprocessing.py : Cleaning, normalizing, tokenizing, etc.
- src/features/
  - feature_engineering.py
- src/models/
  - models.py : Model definitions.
  - train.py : Training entry point.
  - evaluate.py : Evaluation and metrics.
  - predict.py : Inference script.
- src/visualization/
  - plotting.py
- src/utils/
  - helpers, config loaders, logging setup : Helper functions.
  - plotting.py      : Visualization helpers.
  - metrics.py       : Custom metrics.
  - logging_utils.py : Logging helpers.
  - file_utils.py    : File and path utilities.
  - seed.py          : Reproducibility functions.

### Notes:
- All important logic should live here, not inside notebooks.