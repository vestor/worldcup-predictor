# Worldcup Predictor
Tries to predict a cricket ODI match outcome with minimal data

## How to use?
1. `python worldcup-predictor.py`
2. To modify the current fixture, just edit the variable `match_data` in the file.

## How does it work?
Creates a csv file of historical data from 1990 about the ODIs played. The current data only contains "date", "team1", "team2", "winner", "ground" and "margin". The data is scraped from ESPNCricinfo's website.

Once the data is downloaded, python's `pandas` library is used to clean it up a bit (converting the datatypes to int64). Post that, we train the `RandomForestClassifier` model with the given data split between `train` and `test` components.

The trained model can then be used to predict the outcome of a match with teams and the ground provided. The current real-world accuracy in prediction is around an abysmal 52%

## Future TODOs:
1. Get more data-points for training. For eg. types of players, bowlers, fielders, batsmen etc. and bring about a better accuracy number

2. A real-time prediction model, get the live updates of an ODI and modify/strengthen the accuracy figures.

3. Turn it into a web-app?
