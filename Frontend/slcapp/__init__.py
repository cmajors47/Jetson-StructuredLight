from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory, session, g
from slcapp.config import Config

def create_app(test_config=None):
  app = Flask(__name__, instance_relative_config=None)
  app.config.from_object(Config)
  
  with app.app_context():
    from slcapp import views

  return app
