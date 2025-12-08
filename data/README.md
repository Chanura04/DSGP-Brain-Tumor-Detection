# DATA FOLDER

### Purpose:
Stores all datasets used in the project.

### Subfolders:
- raw : Original, untouched data (never edit).
- interim : Cleaned but not final
- processed : Cleaned and transformed data, final datasets used for modeling.

### Notes:
- Do **NOT** modify files inside the raw/ folder.
- Large files should be excluded from version control.
- Never commit large raw files—use *.gitignore*
- Recommend using DVC, Git LFS, or Kaggle CLI if datasets are large