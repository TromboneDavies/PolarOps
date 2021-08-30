from minions import collector

@collector.route("/")
def index():
    return "Hello, Minions!"
