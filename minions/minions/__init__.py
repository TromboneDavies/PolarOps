
from flask import Flask
collector = Flask(__name__)
collector.secret_key = b'askfjalkj'
from minions import routes
