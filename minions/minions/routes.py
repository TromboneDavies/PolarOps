from flask import render_template, session, redirect, request, url_for
import sqlite3
from minions import collector


@collector.route("/")
@collector.route("/collect")
def collect():
    if 'name' in session:
        conn = sqlite3.connect("pile.db")
        results = conn.execute("select * from prac").fetchall()
        conn.close()
        return render_template("tag.html",results=results)
    else:
        return redirect(url_for("who"))


@collector.route("/who", methods=['GET','POST'])
def who():
    if request.method == "POST":
        if request.form['codename'] != "umwpolarops":
            return """
            <h1 style="color:red;">JUST WHO DO YOU THINK YOU ARE???</h1>
            """
        session['name'] = request.form['name']
        return redirect(url_for("collect"))
    else:
        return render_template("who.html")

