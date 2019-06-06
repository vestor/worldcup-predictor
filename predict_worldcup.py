from __future__ import print_function

import csv
from os import path
import pandas as pd
import requests as requests
from bs4 import BeautifulSoup

from sklearn.model_selection import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np


def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors], data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print('Accuracy : %s' % '{0:.3%}'.format(accuracy))

    kf = KFold(n_splits=5).split(data)

    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train, :])

        train_target = data[outcome].iloc[train]

        model.fit(train_predictors, train_target)

        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))

    model.fit(data[predictors], data[outcome])


def scrape_data():
    for year in range(1990, 2020):
        pAFe_link = 'http://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;id=' + str(
            year) + ';type=year'
        pAFe_response = requests.get(pAFe_link, timeout=5)
        soup = BeautifulSoup(pAFe_response.content, "html.parser")
        table = soup.find("table", attrs={"class": "engineTable"})

        # The first tr contains the field names.
        headings = [th.get_text() for th in table.find("tr").find_all("th")]

        with open('odi_data.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Team1', 'Team2', 'Winner', 'Margin', 'Ground', 'Date', 'Id'])
            for row in table.find_all("tr")[1:]:
                dataset = []
                for td in row.find_all("td"):
                    dataset.append(td.get_text())
                writer.writerow(dataset)

        csvFile.close()


if not path.isfile('odi_data.csv'):
    scrape_data()

world_cup_teams = ['Afghanistan', 'Australia', 'Bangladesh', 'England', 'India', 'New Zealand', 'Pakistan',
                   'South Africa', 'Sri Lanka', 'West Indies']

odi_data = pd.read_csv('odi_data.csv')
odi_data.head()
df_teams = odi_data[odi_data['Team1'].isin(world_cup_teams)
                    & odi_data['Team2'].isin(world_cup_teams)
                    & ~odi_data['Winner'].isin(['-'])]

ground_dict = {}
grounds = df_teams.Ground.unique()
counter = 0
for ground in grounds:
    ground_dict[ground] = counter
    counter = counter + 1

print(ground_dict)
df_teams = df_teams.drop(columns=['Margin', 'Date', 'Id'])

df_teams.replace(['Afghanistan', 'Australia', 'Bangladesh', 'England', 'India',
                  'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka',
                  'West Indies']
                 , ['AF', 'AUS', 'BG', 'ENG', 'IN', 'NZ', 'PAK', 'SA', 'SL', 'WI'], inplace=True)

encode = {'Team1': {'AF': 1, 'BG': 2, 'AUS': 3, 'ENG': 4, 'IN': 5, 'NZ': 6, 'PAK': 7, 'SA': 8, 'SL': 9, 'WI': 10},
          'Team2': {'AF': 1, 'BG': 2, 'AUS': 3, 'ENG': 4, 'IN': 5, 'NZ': 6, 'PAK': 7, 'SA': 8, 'SL': 9, 'WI': 10},
          'Winner': {'AF': 1, 'BG': 2, 'AUS': 3, 'ENG': 4, 'IN': 5, 'NZ': 6, 'PAK': 7, 'SA': 8, 'SL': 9, 'WI': 10,
                     'no result': 11, 'tied': 12},
          'Ground': ground_dict
          }

winner_dict = {'AF': 1, 'BG': 2, 'AUS': 3, 'ENG': 4, 'IN': 5, 'NZ': 6, 'PAK': 7, 'SA': 8, 'SL': 9, 'WI': 10,
               'no result': 11, 'tied': 12}
df_teams.replace(encode, inplace=True)

df = pd.DataFrame(df_teams)
model = RandomForestClassifier(n_estimators=100)
outcome_var = ['Winner']
predictor_var = ['Team1', 'Team2', 'Ground']
classification_model(model, df, predictor_var, outcome_var)

match_data = [['AF', 'SL', 'Cardiff']]

for data in match_data:
    team1 = data[0]
    team2 = data[1]
    input = [winner_dict[team1], winner_dict[team2], ground_dict[data[2]]]
    input = np.array(input).reshape((1, -1))
    output = model.predict(input)
    print(list(winner_dict.keys())[list(winner_dict.values()).index(output)])
