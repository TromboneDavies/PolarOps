
from flask import Flask
collector = Flask(__name__, instance_relative_config=True)
collector.config.from_envvar("SETTINGS_FILE")
collector.secret_key = collector.config['SECRET_KEY']
from minions import routes
