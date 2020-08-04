# Metronome
[![Build Status](https://travis-ci.org/kpagacz/glyculator-cleaner.svg?branch=app)](https://travis-ci.org/kpagacz/glyculator-cleaner)
[![Maintainability](https://api.codeclimate.com/v1/badges/5bcae2030c59f8863739/maintainability)](https://codeclimate.com/github/kpagacz/glyculator-cleaner/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/5bcae2030c59f8863739/test_coverage)](https://codeclimate.com/github/kpagacz/glyculator-cleaner/test_coverage)

## Introduction
Metronome is a very simple python app designed to find a pattern in a series of time points. 
Given a set of differences between consecutive date and time-described time points 
and designated frequency it tries to fit the data into the pattern of time points repeating in the designated frequency.

Underneath the hood the predictions are handled by a tensorflow.keras neural network. The web engine is setup
in Flask. Metronome currently does not have a production-grade hosting engine attached to it. The server packaged
with the app is the one bundled with Flask - Werkzeug and it is not dedicated to production.

## Metronome setup
### Requirements
Metronome requires Python version 3.4+.

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

Metronome listens on port 5000 on default, but it can be modified in cleaner.py by changing the port
argument:
```python
app.run(port=8080)
```
After such a change Metronome will listen on port 8080.

## Metronome API