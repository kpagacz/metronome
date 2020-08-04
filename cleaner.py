import flask 
import numpy as np
import json

from metronome.Metronome import Metronome5
from utils.app_utils import NumpyEncoder

# TO-DO (konrad.pagacz@gmail.com) expand docs - a solid readme is much needed
# TO-DO (konrad.pagacz@gmail.com) add logging functionalities

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Cleaner setup
cleaner_interval_5 = Metronome5()

@app.route("/v1/models/metronome", methods=["POST"])
def clean():
    incoming = flask.request.get_json()
    incoming_data = json.loads(incoming)

    cleaner_type = incoming_data.pop("interval")

    if(cleaner_type == 5):
        model_output = {
            "probabilities" : list(cleaner_interval_5.predict_proba(incoming_data)),
            "predictions" : list(cleaner_interval_5.predict(incoming_data)),
        }
        return json.dumps(model_output, cls=NumpyEncoder)
    else:
        return json.dumps({})
    

app.run(port=5000)
