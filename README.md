Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Start by setting DagsHub as your distant storage through DVC.

```bash
dvc init
dvc remote add -f origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/aryan.absalan/RoadAccidentsInFrance.s3 
dvc remote default origin
```
Use dvc to connect the different steps of your pipeline.

The command for addind all of the steps of the pipeline are: 

```bash
dvc stage add -n data_ingestion -d src/pipeline_steps/stage01_data_ingestion.py -d src/config.yaml -o data/raw/accidents_data_red.csv python src/pipeline_steps/stage01_data_ingestion.py

dvc stage add -n data_validation -d src/pipeline_steps/stage02_data_validation.py -d data/raw/accidents_data_red.csv -o data/validated/accidents_data_validated.csv python src/pipeline_steps/stage02_data_validation.py

dvc stage add -n data_transformation -d src/pipeline_steps/stage03_data_transformation.py -d data/validated/accidents_data_validated.csv -o data/transformed/accidents_data_transformed.csv python src/pipeline_steps/stage03_data_transformation.py


dvc stage add -n model_training -d src/pipeline_steps/stage04_model_trainer.py -d data/transformed/accidents_data_transformed.csv -o models/DecisionTreeClassifier_model.pkl python src/pipeline_steps/stage04_model_trainer.py

dvc stage add -n model_evaluation -d src/pipeline_steps/stage05_model_evaluation.py -d models/DecisionTreeClassifier_model.pkl -d data/validated/accidents_data_validated.csv -o reports/model_evaluation_report.txt python src/pipeline_steps/stage05_model_evaluation.py

```
You can run the pipeline through the command `dvc repro`.

dvc add data/raw/accidents_data.csv
dvc add data/validated/accidents_data_validated.csv
dvc add data/transformed/accidents_data_transformed.csv
dvc add models/DecisionTreeClassifier_model.pkl
dvc add reports/decision_tree_evaluation_report.txt

dvc commit data/raw/accidents_data.csv
dvc push


To track the changes with git, run: git add dvc.lock 'reports\.gitignore'
To enable auto staging, run: dvc config core.autostage true
Use `dvc push` to send your updates to remote storage.