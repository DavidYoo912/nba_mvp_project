# NBA MVP Prediction Project
![](https://github.com/davidyoo912/nba_mvp_project/misc/mvp_trophy.jpeg?raw=true)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![last commit](https://img.shields.io/github/last-commit/davidyoo912/nba_mvp_project?color=orange)
![pull requests](https://img.shields.io/github/issues-pr/davidyoo912/nba_mvp_project)

In this project, the objective is to build a NBA MVP prediction model and forecast the upcoming MVP for the current (2022) season. 

To run the notebook files using the required dependencies, simply create a new environment using the yml file with conda
```
conda env create -f environment.yml
```
then switch to the **nba_mvp_project** kernel

## Data
A combination of pandas HTML table scraping function and this basketball reference scraping tool were utilized to pull raw data.
CSV files have been saved in this folder

## Notebook
Notebook folder contains the main notebook for the analysis **nba_mvp_prediction.ipynb** as well as the notebook for parameter searching **parameter_tuning.ipynb**
