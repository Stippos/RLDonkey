import sqlite3
import datetime

def get_connection(db):
    conn = sqlite3.connect(db)
    return conn, conn.cursor()

def create_db(db_name="DB_{}.db".format(datetime.datetime.now().isoformat())):

    conn, c = get_connection(db_name)

    c.execute("CREATE TABLE sweeps (id integer primary key, time text, conf text, description text)")

    c.execute("CREATE TABLE sessions (id integer primary key, time text, description text, model_name text)")

    c.execute("CREATE TABLE params (session int, name text, value real)")

    c.execute("CREATE TABLE episodes (session int, episode int, time text, steps int, reward int)")
    conn.commit()

    conn.close()

def insert_sweep(db, conf, description):

    conn, c = get_connection(db)

    t = datetime.datetime.now().isoformat()

    c.execute("INSERT INTO sweeps (id, time, conf, description) VALUES (NULL, ?, ?, ?)", (t, conf, description))
    ret = c.execute("SELECT id FROM sweeps WHERE time = ?", (t, )).fetchone()[0]
    conn.commit()
    conn.close()

    return ret

def insert_session(db, model_name, description="Failed"):

    conn, c = get_connection(db)
    t = datetime.datetime.now().isoformat()

    c.execute("INSERT INTO sessions (id, time, description) VALUES (NULL, ?, ?)", (t, description))

    ret = c.execute("SELECT id FROM sessions WHERE time = ?", (t,)).fetchone()[0]
    conn.commit()
    conn.close()

    return ret

def insert_parameters(db, session, params):

    conn, c = get_connection(db)

    vals = [(session, x[0], x[1]) for x in params]

    c.executemany("INSERT INTO params (session, name, value) VALUES (?, ?, ?)", vals)

    conn.commit()
    conn.close()

def insert_episode(db, session, episode, steps, reward):

    conn, c = get_connection(db)

    t = datetime.datetime.now().isoformat()

    c.execute("INSERT INTO episodes (session, episode, time, steps, reward) VALUES (?,?,?,?,?)",
    (session, episode, t, steps, reward))

    conn.commit()
    conn.close()

def update_description(db, session, status):

    conn, c = get_connection(db)

    c.execute("UPDATE sessions SET description = ? WHERE id = ?", (status, session))
    
    conn.commit()
    conn.close()