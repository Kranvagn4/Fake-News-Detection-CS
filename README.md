# Fake News Detection Using NLP

A Python-based machine learning project that classifies news articles as **Fake** or **Real** using text preprocessing, TF-IDF feature extraction, and multiple classifiers.

**GitHub profile:**  https://github.com/Kranvagn4/
**Repository URL:** Add the URL of this project repository after creating it on GitHub.

> **Disclaimer:** This project identifies text patterns learned from a dataset. Its predictions are not independent fact-checking and should not be treated as proof that a news story is true or false.

## Features

- Loads separate fake-news and real-news CSV datasets.
- Combines the article title and text when those columns are available.
- Cleans text by removing HTML tags, URLs, non-alphabetic characters, and extra whitespace.
- Uses TF-IDF features with unigrams and bigrams.
- Trains and compares:
  - Logistic Regression
  - Linear Support Vector Machine (Linear SVM)
  - XGBoost
- Reports accuracy, precision, recall, F1-score, classification report, confusion matrix, and ROC-AUC where supported.
- Displays exploratory plots, including class distribution, word clouds, model comparison, and ROC curves.
- Selects the model with the highest test-set F1-score.
- Saves the selected model and TF-IDF vectorizer together in a Joblib artifact.
- Provides an interactive prompt to classify custom news text.

## Project Files

The supplied ZIP contains the following files:

```text
project_cs/
├── project       # Python source script; rename to main.py for convenience
├── fake.csv      # Fake-news dataset
└── true.csv      # Real-news dataset
```

For a clean GitHub repository, use this recommended structure:

```text
Fake_News_Detection_NLP/
├── README.md
├── main.py
├── requirements.txt
├── data/
│   ├── fake.csv
│   └── true.csv
└── fake_news_detector.joblib  # generated after successful training
```

The model artifact is generated when the script completes training; it is not created simply by downloading the source files.

## Dataset

The project uses two CSV files:

- `fake.csv` — records labelled as Fake by the script.
- `true.csv` — records labelled as Real by the script.

The inspected files contain these columns:

- `title`
- `text`
- `subject`
- `date`

The script requires a `text` column. It uses `title` if present and combines the title and article body for classification. The `subject` and `date` columns are retained in the combined dataframe but are not used as model features by the current implementation.

Keep the two CSV files in the same directory as the script when using the default commands below, or pass their locations with `--fake` and `--true`.

Before redistributing the dataset publicly, confirm the dataset's license and attribution requirements.

## Requirements

- Python 3.9 or a compatible Python version supported by the installed dependency releases
- `pip`
- Sufficient RAM and disk space to load and process the CSV files and train the models

The source uses these Python packages:

- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- WordCloud

## Installation

### 1. Download or clone the repository

After creating the repository on GitHub, clone it with the actual repository URL:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create a virtual environment

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` containing the packages used by the script:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
wordcloud
```

Then install them:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a reproducible academic submission, test the project with a compatible environment and record the working package versions.

## Run the Project

Rename the supplied source file `project` to `main.py` (or use its existing filename in the commands).

Place `fake.csv` and `true.csv` in the same directory as `main.py`.

Run:

```bash
python main.py
```

The default command expects files named `fake.csv` and `true.csv` in the current working directory and writes the model artifact to `fake_news_detector.joblib`.

### Specify dataset and output paths

The script supports command-line arguments:

```bash
python main.py --fake data/fake.csv --true data/true.csv --output models/fake_news_detector.joblib
```

Create the `data/` and `models/` directories before running that example, or adjust the paths to match your folder structure.

## What Happens During Execution?

1. The two CSV files are loaded and labelled.
2. Rows with empty combined title/body text are removed, and duplicate title/body records within the same class are dropped.
3. Text is cleaned.
4. Data is split into training and test sets using an 80/20 split, `random_state=42`, and stratification.
5. TF-IDF features are fitted on training text and then applied to test text.
6. Logistic Regression, Linear SVM, and XGBoost are trained.
7. The models are compared using test-set metrics.
8. The model with the highest test-set F1-score is selected.
9. Evaluation plots and reports are displayed.
10. The selected model and vectorizer are saved together in a Joblib artifact.
11. The program asks whether you want to enter a custom article for classification.

## Output

The script saves the selected model and fitted vectorizer together in the path supplied with `--output` (by default, `fake_news_detector.joblib`).

The current source displays evaluation plots and reports during execution. It does not currently save every plot or report to an `outputs/` folder. Add explicit `savefig`/file-writing logic if persistent output files are required.

## Prediction Interpretation

The script prints a predicted label of `Fake` or `Real`. For a model that implements `predict_proba`, it also prints the model's probability for the Fake class. Linear SVM does not provide `predict_proba` by default, and its decision score must not be interpreted as a calibrated probability.

A prediction may be wrong because of dataset bias, changes in writing style, unfamiliar topics, misleading text, or other limitations. Verify important claims using reliable independent sources.

## Troubleshooting

**`FileNotFoundError`**  
Make sure both CSV files exist at the paths passed to `--fake` and `--true`.

**`ValueError: Both CSVs must include a 'text' column`**  
Check the CSV headers. The current implementation requires a `text` column in both files.

**`ModuleNotFoundError`**  
Activate the intended virtual environment and run `pip install -r requirements.txt`.

**XGBoost installation or compatibility error**  
Use a Python version and XGBoost release compatible with your operating system. Consult the official XGBoost installation documentation if needed.

**The run takes a long time or uses substantial memory**  
The script trains three classifiers on TF-IDF features. Try a machine with more available memory, close other applications, or test with a documented subset during development. Do not report subset results as full-dataset results.

## Current Limitations

- The application is a command-line script; it does not include a web interface or desktop GUI.
- The source displays plots interactively rather than saving all figures automatically.
- The script requires the dataset CSV files for training.
- The current implementation uses a single train/test split for model comparison.
- The classifier does not independently verify sources, citations, or the factual accuracy of claims.
- A trained model file is generated only after successful execution.

## GitHub Upload

1. Create a new repository on GitHub under your account.
2. Add `README.md`, the Python source, and `requirements.txt`.
3. Add the dataset only after confirming that redistribution is permitted and that repository file-size limits are respected.
4. Commit and push the files using Git, or upload them through GitHub's web interface.
5. Replace the repository placeholder near the top of this README with the actual repository URL.

Example Git commands (run from the project directory after creating an empty GitHub repository):

```bash
git init
git add README.md main.py requirements.txt
git commit -m "Add Fake News Detection NLP project"
git branch -M main
git remote add origin YOUR_ACTUAL_REPOSITORY_URL
git push -u origin main
```

If your project already has a Git repository or remote, inspect its configuration before running `git init` or adding a remote.

## License

Add a project license only after choosing appropriate terms. The code license and dataset license may be different; verify the dataset's source and redistribution conditions before publishing the CSV files.
