from flask import ( render_template, session, redirect, request, url_for,
    current_app, Markup)
import sqlite3
import os
from minions import collector

all_piles_users = ['sabine','ryan']
pile_assignments = {
    'christine':1,
    'ajay':1,
    'stephen':3,
    'veronica':3,
    'tj':2,
    'alexis':2,
}

@collector.route("/")
@collector.route("/collect", methods=['GET','POST'])
def collect():
    if 'name' in session:
        if request.method == 'POST':
            record_tag(request.form['cid'], session['name'],
                request.form['rating'])
            return redirect(url_for("collect"))
        else:
            cid, thread, num_threads = select_thread_for(session['name'])
            return render_template("tag.html",thread_to_rate=Markup(thread),
                num_threads=num_threads+1, cid=cid)
    else:
        return redirect(url_for("who"))


# Returns a tuple containing the comment ID of a thread, the thread's text, and
# the number of threads already tagged by this user.
def select_thread_for(name):
    conn = sqlite3.connect(os.path.join(collector.instance_path,
            current_app.config['DATABASE']),
        detect_types=sqlite3.PARSE_DECLTYPES)
    if name in ['pile','rated']:
        print(f"NO WAY will we allow '{name}' as a name!")
        return 0
    conn.execute(
        f"""
        drop table if exists {name}
        """
    )
    conn.execute(
        f"""
        create temporary table {name} as 
            select comment_id from pile
            except select comment_id from rated where rater=?
        """, (name,)).fetchone()
    thread_to_rate = conn.execute(
        f"""
        select comment_id,text from pile where comment_id=
        (select comment_id from {name} order by random() limit 1)
        """).fetchone()
    num_threads = conn.execute(
        f"""
        select count(*) from rated where rater=?
        """, (name,)).fetchone()[0]
    conn.close()
    thread = thread_to_rate[1].replace("\\n","<br>")
    return thread_to_rate[0], thread, num_threads


# Store this user's tag for the comment ID passed.
def record_tag(comment_id, name, tag):
    conn = sqlite3.connect(os.path.join(collector.instance_path,
            current_app.config['DATABASE']),
        detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute(
        f"""
        insert into rated (comment_id, rater, rating) values (?,?,?)
        """,
        (comment_id, name, tag)
    )
    conn.commit()
    conn.close()


@collector.route("/who", methods=['GET','POST'])
def who():
    if request.method == "POST":
        if request.form['codename'] != "umwpolarops":
            return """
            <h1 style="color:red;">JUST WHO DO YOU THINK YOU ARE???</h1>
            """
        if (request.form['name'] not in all_piles_users and
            request.form['name'] not in pile_assignments):
            return """
            <h2 style="color:red;">
            Invalid username: use your first name, all lower-case, please!</h2>
            <p>Press the browser's 'back' button to try again.</p>
            """
        session['name'] = request.form['name']
        return redirect(url_for("collect"))
    else:
        return render_template("who.html")

