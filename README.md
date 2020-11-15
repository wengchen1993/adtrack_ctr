# adtrack_ctr
An example ETL repo for AdTrack CTR application.

## To Run:
```buildoutcfg
pip3 install -r requirements.txt
python3 setup.py install
python3 etl.py --data_source "./data" --model_name "xgb" --num_sampled_data=1500000
```
This setup will takes about 10 minutes to generate the model artefact OR   
refer [here](./logs/etl.log) to read the logs OR  
refer [here](./src/notebooks/eda.ipynb) for Jupyter Notebook ran results.


## Dataset
Dataset used is retrieved from the Kaggle competition - **TalkingData AdTracking Fraud Detection Challenge** .  
You can get it from [here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data?select=train.csv) .

## Project structure 
```buildoutcfg
.
|- artefacts  (Store model artefacts)
|- data  (Store all data files, use external DB in prod)
  |- train
    |- train.csv
|- logs  (Model training process logs)
|- src
  |- evaluation
    |- metrics.py  (Contains all evaluation metric logic)
    |- model_performance.py  (Main evaluation performance runner)
  |- notebooks
    |- eda.ipynb  (Core logic of feature engineering and model training)
    |- eda.md  (Jupytext Markdown equivalent for code change tracking)
  |- pipeline
    |- main.py  (Combine both preprocessors and model as an unified ML Flow pipeline)
    |- model.py  (Model Pipeline step)
    |- preprocess.py  (Preprocessing Pipeline step)
  |- preprocessing
    |- features_generators  (Preprocessing and feature generation logic)
      |- categorical  (Contains all categorical feature generator)
      |- datetime  (Contains all datetime feature generator)
      |- extract  (Acts as a sub-columns selector to pass into other feature generators)
  |- training
    |- models
      |- logit.py  (Logistic Regression logic and Grid Search parameters)
      |= xgboost.py  (XGBoost logic and Grid Search parameters)
|- etl.py  (Main ETL executor, contains highest level of logic of the steps)
|- requirements.txt  (all pip dependencies specification)
|- setup.py  (Module setup)
```
