## DendroMeter Gap Filling with Extreme Gradient Boosting

  - [Requirements](#requirements)
  - [Functions](#content)


---

## Requirements

- Python (>=3.9)
- Conda ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)), or [Pip](https://pip.pypa.io/en/stable/installation/)
- Visual Studio Code [VSCode](https://code.visualstudio.com/)

The code is based on the following packages `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`, `scikit-optimize`, `pyrealm` and `xgboost` .

So far, code has been tested on Windows10. 



### Data Preparation

Data should be cleaned and the zero-growth model (Zweifel et al. 2016, Knüsel et al. 2021) should have been applied. The data should be in a dataframe in hourly resolution consitsing of te following columns: 
'DOY': day of year
'year': the years
'hour': hour of the day [0-23]
'Label': output 'GRO' from the zero-growth model. 

Note: the temporal variables should have consecutive data and NO missing values or NAs. the 'Label' should include the NAs, where data gaps are. 

#### Function1  `create_df_csv`

- Description: reads csv files with any kind of separator 
- Options: 
    - `path`: chr, your path to the csv file

#### Function2  `clean data`

- Description: detects gaps and checks for negative values
- Options: 
    - `df`: your dataframe read by create_df_csv contraining the columns 'DOY', 'year', 'hour', 'Label'
- Returns:
    - `cleaned data`: the data, checked for negative values and cleaned from gaps 
    - `data gaps`: the extracted data gap rows

#### Function3  `traintestsplit`

- Description: splits your data in training and test subset
- Options: 
    - `df`: your cleaned dataframe
- Returns:
    - `X_train`: training subset features (DOY,year,hour)
    - `X_test`: test subset features (DOY,year,hour)
    - `y_train`: training subset growth label (Label)
    - `y_test`: test subset growth label (Label)
 
#### Function4  `ZTransformTrain`

- Description: scaling of the data
- Options: 
    - `X_train`: the X training subset features (DOY,year,hour)
    - `path`: your path + name of the scaler (NAME.bin], where you want to save your scaler
- Returns:
    - `X_train_scaled`: scaled training subset features (DOY,year,hour)
    - `Scaler`:the scaler of the training subset


### Modelling

#### Function 5  `testall4`

- Description: Not neccessary for gap filling but it tests the best three algorithms and ridge regression on your data.
- Required parameters: 
    - `X_train_scaled`: your scaled X training subset
    - `X_test_scaled`: the scaled X test subset. Note: this subset should be scaled with the properties of the scaler from the training subset. 
    - `y_train`
    - `y_test`
- Returns:
    - `Performances`: The performances (RMSE and R²) of the different algorithms on the training and test subsets

#### Function 6  `testxgb`

- Description: Fits the extreme gradient boosting (XGB) algorithm to your data and returns the performance, the predictions for training and test subset and the model.
- Required parameters: 
    - `X_train_scaled`: your scaled X training subset
    - `X_test_scaled`: the scaled X test subset. Note: this subset should be scaled with the properties of the scaler from the training subset. 
    - `y_train`
    - `y_test`
- Returns:
    - `Performances`: The performances (RMSE and R²) of the different algorithms on the training and test subsets
    - `Train_predict`: The predictions of the model for the training subset
    - `Test_predict`: The predictions of the model for the test subset
    - `Model`: The xgb regression model, which can be applied to predict the data gaps
--
