import flask 
import numpy as np
import json

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from metronome.Metronome import Metronome5
from utils.app_utils import NumpyEncoder


app = flask.Flask(__name__)

# Cleaner setup
metronome_interval_5 = Metronome5()

@app.route("/v1/models/metronome", methods=["POST"])
def clean():
    # Logger setup
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    
    # Load the incoming json
    incoming = flask.request.get_json()
    incoming_data = json.loads(incoming)
    logger.debug(incoming)

    interval = incoming_data.pop("interval")

    if(type(interval) == int):
        model_output = {
            "probabilities" : list(metronome_interval_5.predict_proba(incoming_data, interval)),
            "predictions" : list(metronome_interval_5.predict(incoming_data, interval)),
        }
        return json.dumps(model_output, cls=NumpyEncoder)
   
    else:
        return json.dumps({})
    

