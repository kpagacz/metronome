import flask 
import numpy as np
import json
import logging


from cleaner.Cleaner import Cleaner5
from cleaner.config import WINDOW_SIZE
from utils.app_utils import NumpyEncoder


app = flask.Flask(__name__)
app.config["DEBUG"] = True

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Cleaner setup
cleaner_interval_5 = Cleaner5()

@app.route("/v1/models/cleaner", methods=["POST"])
def clean():
    incoming_data = {
        "var" + str(i) : np.float(flask.request.form["var" + str(i)]) for i in range(WINDOW_SIZE - 1)
    }

    cleaner_type = np.float(flask.request.form["interval"])

    if(cleaner_type == 5):
        model_output = {
            "probabilities" : list(cleaner_interval_5.predict_proba(incoming_data)),
            "predictions" : list(cleaner_interval_5.predict(incoming_data)),
        }

        return json.dumps(model_output, cls=NumpyEncoder)
    else:
        return json.dumps({})
    

logger.info("Welcome to CGM cleaning services of Konrad Pagacz!")
logger.info("Starting up server")
app.run()
