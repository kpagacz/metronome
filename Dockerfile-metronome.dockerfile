FROM python:3.8

# uWSGI installation
RUN pip install uWSGI

# Requirements
COPY requirements.txt /srv/metronome/

WORKDIR /srv/metronome
RUN pip install -r requirements.txt

# Copy application
COPY . /srv/metronome

EXPOSE 5000
CMD uwsgi uwsgi-config.ini