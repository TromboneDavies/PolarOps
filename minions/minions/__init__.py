
from flask import Flask
collector = Flask(__name__)
from minions import routes
