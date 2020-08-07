from flask import Flask
from flask_pymongo import PyMongo
import json
import os
from random import choice
from bson.json_util import dumps
from flask import request
from datetime import datetime
from flask_cors import CORS, cross_origin
import uuid

app = Flask(__name__)

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# DB connection
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
MONGO_URI = f'mongodb://{username}:{password}@ds137267.mlab.com:37267/win_codenames?retryWrites=false'
app.config["MONGO_URI"] = MONGO_URI
app.config["connectTimeoutMS"] = 30000
app.config['socketTimeoutMS'] = None
app.config['socketKeepAlive'] = True
app.config['connect'] = False
app.config['maxPoolSize'] = 1
mongo = PyMongo(app)

PREFIX = "/api"

def get_id(id_length=8):
    return str(uuid.uuid4())[:id_length]

def get_games(game_id):
    if game_id:
        game = mongo.db.games.find({"id": game_id})
        return game
    # Get list of all game ids
    game_ids = mongo.db.games.find().distinct('id')
    # Choose one at random
    random_game_id = choice(game_ids)
    # Get that random game and return
    return mongo.db.games.find({"id": random_game_id})

@app.route(PREFIX + '/games/', defaults={'game_id': None})
@app.route(PREFIX + '/games/<game_id>/')
def games(game_id):
    game = get_games(game_id)
    return dumps(game)

@app.route(PREFIX + '/reviews/')
def reviews():
    return 'reviews'

@app.route(PREFIX + '/create-review/', methods=['POST'])
def create_review():
    req = json.loads(request.get_data())
    req['id'] = get_id()
    req['created'] = datetime.now()
    res = mongo.db.reviews.insert_one(req)
    return 'great'
