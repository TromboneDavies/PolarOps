from flask import ( render_template, session, redirect, request, url_for,
    current_app, Markup)
import sqlite3
import os
from minions import collector


@collector.route("/")
@collector.route("/collect")
def collect():
    if 'name' in session:
        conn = sqlite3.connect(os.path.join(collector.instance_path,
                current_app.config['DATABASE']),
            detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute(
            f"""
            drop table if exists {session['name']}
            """
        )
        conn.execute(
            f"""
            create temporary table {session['name']} as 
                select comment_id from pile
                except select comment_id from rated where rater=?
            """, (session['name'],)).fetchone()
        thread_to_rate = conn.execute(
            f"""
            select comment_id,text from pile where comment_id=
            (select comment_id from {session['name']} order by random() limit 1)
            """).fetchone()
        num_threads = conn.execute(
            f"""
            select count(*) from rated where rater=?
            """, (session['name'],)).fetchone()[0]
        conn.close()
        thread = thread_to_rate[1].replace("\\n","<br>")
        return render_template("tag.html",thread_to_rate=Markup(thread),
            num_threads=num_threads+1)
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

