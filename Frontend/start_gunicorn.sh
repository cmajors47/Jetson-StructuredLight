#!/bin/bash
cd Jetson-StructuredLight/Frontend
gunicorn --config slcapp/gunicorn_config.py "slcapp:create_app()"
