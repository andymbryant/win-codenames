from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.pipeline import Pipeline
from datetime import datetime
from joblib import dump
import pickle
import json
import numpy as np
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from glob import glob

dirname = os.path.dirname(__file__)

load_dotenv()

DOWNLOAD_REVIEWS = False
DOWNLOAD_GAMES = False
SAVE_GAMES = False
SAVE_REVIEWS = False
TUNE_PARAMS = False
SAVE_MODEL = True

now = datetime.now()
time_info = now.strftime("%m-%d-%H-%M")

y_column = 'human_selected'
X_columns = ['rank', 'goodness', 'bad_minimax', 'frequency', 'neutrals_minimax', 'variance']
y_var = 'human_selected'
y_column = [y_var]
all_columns = X_columns + y_column
no_good_clues = 0
output = pd.DataFrame(columns=all_columns)

if DOWNLOAD_REVIEWS or DOWNLOAD_GAMES:
    username = os.getenv('MONGO_USERNAME')
    password = os.getenv('MONGO_PASSWORD')
    MONGO_URI = f'mongodb://{username}:{password}@ds137267.mlab.com:37267/win_codenames?retryWrites=false'
    client = MongoClient(MONGO_URI)
    db = client.win_codenames

if DOWNLOAD_REVIEWS:
    print('Connected to database...')
    all_reviews = [review for review in db.reviews.find()]
    if SAVE_REVIEWS:
        json_filepath = os.path.join(dirname, f'output/reviews_pkl/reviews_{time_info}.pkl')
        with open(json_filepath, 'wb') as outfile:
            pickle.dump(all_reviews, outfile)
else:
    reviews_path = os.path.join(dirname, 'output/reviews_pkl/*')
    list_of_files = glob(reviews_path)
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'rb') as pickle_file:
        all_reviews = pickle.load(pickle_file)

if DOWNLOAD_GAMES:
    all_games = []
    for review in all_reviews:
        game_id = review['game_id']
        game = dict(db.games.find_one({'id': game_id}))
        all_games.append(game)
    if SAVE_GAMES:
        json_filepath = os.path.join(dirname, f'output/games_pkl/games_{time_info}.pkl')
        with open(json_filepath, 'wb') as outfile:
            pickle.dump(all_games, outfile)
else:
    games_path = os.path.join(dirname, 'output/games_pkl/*')
    list_of_files = glob(games_path)
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'rb') as pickle_file:
        all_games = pickle.load(pickle_file)

num_reviews = len(all_reviews)
all_reviewers = list(set([review['reviewer'] for review in all_reviews]))
num_reviewers = len(all_reviewers)
print(f'Processing {num_reviews} reviews from {num_reviewers} reviewers...')

i = -1
for review in all_reviews:
    game_id = review['game_id']
    game = [game for game in all_games if game['id'] == game_id][0]
    for clue in game['clues']:
        i += 1
        new_row = clue
        new_row[y_var] = 1 if review[y_var] == clue['word'] else 0
        output.loc[i] = new_row
    if review['human_selected'] == 'no_good_clues':
        no_good_clues += 1

output_path = os.path.join(os.path.dirname(__file__), f'output/results/results_{time_info}.csv')
output.to_csv(output_path)

num_top_selected = output.human_selected.sum()
print(num_top_selected)
print(num_reviews)
print(f'Humans and computers selected the same top word {int((num_top_selected/output.shape[0])*100)}% of the time.')
print(f'Humans thought there were no good clues {int((no_good_clues/num_reviews)*100)}% of the time.')

print('Preparing data...')
X = output[X_columns].values.tolist()
y = output[y_var].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Very helpful read
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
if TUNE_PARAMS:
    grid_params = {
        'C': np.linspace(0.0001,2,21),
        'epsilon': np.linspace(0,1,20),
        'gamma': np.linspace(0,2,20)
    }

    cv = 5
    model = SVR()
    print('Start search...')
    search = GridSearchCV(model, param_grid=grid_params, cv=cv)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f'Best params: {search.best_params_}')
    C = best_params['C']
    epsilon = best_params['epsilon']
    gamma = best_params['gamma']
    # Multiple iterations of search
    # Best params: {'C': 0.32435022868815844, 'epsilon': 0.1, 'gamma': 2.0, 'kernel': 'rbf'}
    # Best params: {'C': 0.32435022868815844, 'epsilon': 0.08163265306122448, 'gamma': 2.0, 'kernel': 'rbf'}
    # Best params: {'C': 0.2378613016572934, 'epsilon': 0.10526315789473684, 'gamma': 'auto'}
    # Best params: {'C': 0.300085, 'epsilon': 0.10526315789473684, 'gamma': 1.0}
    # Best params {'C': 0.300085, 'epsilon': 0.10526315789473684, 'gamma': 0.8421052631578947}
else:
    C = 0.3
    epsilon = 0.1
    gamma = 0.00001

model = SVR(C=C, epsilon=epsilon, gamma=gamma)
print('Fitting model...')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R2: {r2}, MSE: {mse}')

if SAVE_MODEL:
    print('Saving model...')
    model_output_path = os.path.join(dirname, f'output/models/svr_{time_info}.joblib')
    dump(model, model_output_path)


