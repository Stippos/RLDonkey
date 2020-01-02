import sqlite3
import datetime

def get_connection(db):
    conn = sqlite3.connect(db)
    return conn.cursor()

def create_db(hyperparameters, db_name="DB_{}.db".format(datetime.datetime.now().isoformat())):

    c = get_connection(db_name)

    c.execute("CREATE TABLE sweeps (id integer primary key, time text, conf text, description text)")
    
    c.excecute("CREATE TABLE sessions (id integer primary key, time text, description text)")
    c.commit()

    c.execute("CREATE TABLE params (session int, name text, value real)")
    c.commit()

    c.execute("CREATE TABLE episodes (session int, episode int, time text, steps int, reward int)")
    c.commit()

    c.close()

def new_session(db, description="Failed"):

    c = get_connection(db)
    t = datetime.datetime.now()

    c.execute("INSERT INTO sessions (id, time, description) VALUES (NULL, ?, ?)", (t, description))

    ret = c.execute("SELECT id FROM sessions WHERE time = ?", (t,)).fetchone()[0]
    c.close()

    return ret

def insert_parameters(db, session, params):

    c = get_connection(db)

    vals = [(session, x[0], x[1]) for x in params]

    c.executemany("INSERT INTO params (session, name, value) VALUES (?, ?, ?)", vals)

    c.commit()
    c.close()

def insert_episode(db, session, episode, time, steps, reward):

    c = get_connection(db)

    c.execute("INSERT INTO episodes (session, episode, time, steps, reward) VALUES (?,?,?,?,?)",
    (session, episode, time, steps, reward))

    c.close()

def update_description(db, session, status):

    c = get_connection(db)

    c.execute("UPDATE sessions SET description = ? WHERE id = ?", (status, session))

    c.close()