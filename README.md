# Metronome
[![Build Status](https://travis-ci.org/kpagacz/glyculator-cleaner.svg?branch=app)](https://travis-ci.org/kpagacz/glyculator-cleaner)
[![Maintainability](https://api.codeclimate.com/v1/badges/c6147e4561478ec28b30/maintainability)](https://codeclimate.com/github/kpagacz/metronome/maintainability)
[![codecov](https://codecov.io/gh/kpagacz/metronome/branch/app/graph/badge.svg)](https://codecov.io/gh/kpagacz/metronome)


## Introduction
Metronome is a very simple python app designed to find a temporal pattern in a series of time points. 
Given a set of differences between consecutive date and time-described time points 
and designated frequency it tries to fit the data into the pattern of time points repeating in the designated frequency.

Underneath the hood the predictions are handled by a tensorflow.keras neural network. The web engine is setup
in Flask. Metronome currently does not have a production-grade hosting engine attached to it. The server packaged
with the app is the one bundled with Flask - Werkzeug and it is not dedicated to production.

## Metronome setup
### Requirements
Metronome requires Python version 3.6+.

### Installation
Using the shell:
```bash
git clone https://github.com/kpagacz/metronome
cd metronome
pip install -r requirements.txt
```

### Running
Using the shell:
```bash
python cleaner.py
```

Once this command is ran Metronome listens on the default port 5000 for icoming HTTP calls. Default port can be modified in cleaner.py file
via port argument to app.run(port=port) call.
```python
app.run(port=8080)
```
After such a change and a restart of the app Metronome will listen on port 8080. You can set the port argument to an arbitrary value.

## Metronome API
The Metronome API allows its users to access the model predictions and probabilities.

It exposes a single resource at /v1/models/metronome/ expecting a POST HTTP request at this endpoint.
The app does not perform any user input validation, so its app to the caller to make sure the calls
follow the structure.

### Minimal working example
Given Metronome runs on localhost:5000 port, one can use it in a following way:
```python
import requests
import json
import numpy as np


example_data = {
    "var" + str(i) : list(np.random.uniform(low=0, high=350)) for i in range(14)
    # API accepts differences in seconds between consecutive time points
}

example_data["interval"] = 5 # interval in minutes
print(example_data)

payload = json.dumps(example_data)
response = requests.post("127.0.0.1:5000/v1/models/metronome", json=payload)
print(response)
```

### /v1/models/metronome
Method: POST
Parameters:
"var0" - list of float values
"var1" - list of float values
"var2" - list of float values
"var3" - list of float values
"var4" - list of float values
"var5" - list of float values
"var6" - list of float values
"var7" - list of float values
"var8" - list of float values
"var9" - list of float values
"var10" - list of float values
"var11" - list of float values
"var12" - list of float values
"var13" - list of float values
"var14" - list of float values 
"interval" - float - number of minutes

Each "var0" - "var14" contains list of float values of seconds between 15 consecutive time points.

### Input requirements
The HTTP post call at /v1/models/metronome needs to have a json payload constructed as follows:
14 keys named from var0, var1, var2 ... to var14. Values have to be a list of differences
between 15 consecutive points in time. Additionaly it has to contain the "interval" key 
with a single float value representing frequency pattern in minutes.
Example of accepted json
```json
{
    "var0": [0],
    "var2": [50],
    "var3": [30],
    "var4": [170],
    "var5": [300],
    "var6": [600],
    "var7": [1],
    "var8": [1000],
    "var9": [450],
    "var10": [750],
    "var11": [20],
    "var12": [66],
    "var13": [900],
    "var14": [129],
    "interval": 5
}
```

My recommendation is to transform python dicts to json via pyjson packge. Example:
```python
import json
import numpy as np 

example_data = {
    "var" + str(i) : list(np.random.uniform(low=0, high=350)) for i in range(14)
    # API accepts differences in seconds between consecutive time points
}

example_data["interval"] = 5 # interval in minutes
payload = json.dumps(example_data)
print(payload)
```

