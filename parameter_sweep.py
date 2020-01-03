import json
import db
import os
import random 

from run_session import run_session

def parameter_sweep(conf_file):

    with open(conf_file) as f:
        conf = json.load(f)
        settings = conf["settings"]
        sweep = conf["sweep"]

    if not os.path.isfile(settings["db"]):
        db.create_db(settings["db"])

    sweep = db.insert_sweep(settings["db"], conf_file, settings["description"])

    for i in range(settings["sessions"]):
        params = {}
        model_name = "sweep_{}_session_{}_model.pth".format(sweep, i)
        session = db.insert_session(settings["db"], model_name)
        for p in sweep:
            param = random.choice(params[p])
            params[p] = param

        db.insert_parameters(settings["db"], session, list(zip(params.keys(), params.values())))

        run_session(settings["db"], sweep, session, model_name, params)

    