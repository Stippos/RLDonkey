import json
import db
import os
import random 

from run_session import run_session

def parameter_sweep(conf_file):

    with open(conf_file) as f:
        conf = json.load(f)
        settings = conf["settings"]

    if not os.path.isfile(settings["db"]):
        db.create_db(settings["db"])

    sweep = db.insert_sweep(settings["db"], conf_file, settings["description"])

    for i in range(settings["sessions"]):
        params = {}
        model_name = "sweep_{}_session_{}_model.pth".format(sweep, i)
        session = db.insert_session(settings["db"], model_name)
        for p in conf["sweep"]:
            param = random.choice(conf["sweep"][p])
            params[p] = param

        db.insert_parameters(settings["db"], session, list(zip(params.keys(), params.values())))
        try:
            run_session(settings["db"], settings["max_session_length"], i, session, model_name, params)
            db.update_description(settings["db"], session, "Success")
        except:
            pass

if __name__ == "__main__":
    parameter_sweep("sweep_3.json")

    