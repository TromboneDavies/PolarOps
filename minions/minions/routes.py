from minions import collector
from flask import render_template, session, redirect, request, url_for

@collector.route("/")
@collector.route("/collect")
def collect():
    if 'name' in session:
        return f"welcome, {session['name']}!"
    else:
        return redirect(url_for("who"))


@collector.route("/who", methods=['GET','POST'])
def who():
    if request.method == "POST":
        if request.form['codename'] != "polarops":
            return """
            <h1 style="color:red;">JUST WHO DO YOU THINK YOU ARE???</h1>
            """
        session['name'] = request.form['name']
        return redirect(url_for("collect"))
    else:
        return render_template("who.html")

