from flask import render_template, request, abort, redirect, url_for, send_file, flash, session, g, current_app, session
import os
import sys
import subprocess

# Import util.py (Normally a simpler process...)
import importlib.util
spec = importlib.util.spec_from_file_location("util", "./slcapp/util.py")
util = importlib.util.module_from_spec(spec)
sys.modules["util"] = util
spec.loader.exec_module(util)

# Global variable to store the subprocess reference
subprocess_ref = None

@current_app.route('/')
def index():
  if 'topLevelFolder' not in session:
    session['topLevelFolder'] = '/home/slc/Jetson-StructuredLight/Frontend'

  if 'activeFileName' not in session:
    session['activeFileName'] = 'No file selected'

  if 'text' not in session:
    session['text'] = 'No file selected'

  #if 'runningProcess' not in session:
  #  session['runningProcess'] = None

  fileTree = util.buildFileTree(session['topLevelFolder'])

  visibleFiles = [{'name': child.name, 'isDir': child.is_directory} for child in fileTree.children]

  return render_template("index.html", rootFolder=session['topLevelFolder'], files=visibleFiles, activeFileName=session['activeFileName'], text=session['text'])

@current_app.route('/fileChange/<filename>')
def changeFileView(filename):
  session['topLevelFolder'] = os.path.join(session['topLevelFolder'], filename)
  return redirect(url_for('index'))

@current_app.route('/up')
def goUpFileView():
  session['topLevelFolder'] = os.path.dirname(session['topLevelFolder'])
  return redirect(url_for('index'))

@current_app.route('/show/<filename>')
def showFileContents(filename):
  session['activeFileName'] = os.path.join(session['topLevelFolder'], filename)

  f = open(os.path.join(session['topLevelFolder'], filename))
  session['text'] = f.read()
  f.close()

  return redirect(url_for('index'))

# Deprecated
@current_app.route('/calibButtonPress', methods=['POST'])
def calibButtonPress():
  if len(request.form.get('cam_resolution')) > 0 and len(request.form.get('graycodeDir')) > 0:
    cam_width = int(request.form.get('cam_resolution'))
    cam_height = int(cam_width * 9 / 16)

    graycodeDir = request.form.get('graycodeDir')

    global subprocess_ref
    print('Hello button press!')

    session['runningProcess'] = 'CalibrationMain.py'
    subprocess_ref = subprocess.Popen(['python3', '/home/slc/Jetson-StructuredLight/Backend2/CalibrationMain.py'])
    print('CalibrationMain.py - done')
    session['runningProcess'] = None
  
    flash("Scripts have been executed!")
  else:
    flash("Camera Resolution or Gray Code Directory not provided.")
  return redirect(url_for('index'))


# Deprecated
@current_app.route('/scanButtonPress', methods=['POST'])
def scanButtonPress():
  if len(request.form.get('cam_resolution')) > 0 and len(request.form.get('graycodeDir')) > 0:
    cam_width = int(request.form.get('cam_resolution'))
    cam_height = int(cam_width * 9 / 16)

    graycodeDir = request.form.get('graycodeDir')

    global subprocess_ref
    print('Hello button press!')

    session['runningProcess'] = 'ScanMain.py'
    subprocess_ref = subprocess.Popen(['python3', '/home/slc/Jetson-StructuredLight/Backend2/ScanMain.py', cam_width, cam_height, graycodeDir])
    print('ScanMain.py - done')
    session['runningProcess'] = None
  
    flash("Scripts have been executed!")
  else:
    flash("Camera Resolution or Gray Code Directory not provided.")
  return redirect(url_for('index'))

@current_app.route('/submit', methods=['POST'])
def submit():
  if len(request.form.get('cam_resolution')) > 0 and len(request.form.get('graycodeDir')) > 0:
    cam_width = int(request.form.get('cam_resolution'))
    cam_height = int(cam_width * 9 / 16)
    graycodeDir = request.form.get('graycodeDir')

    action = request.form['action']
    
    global subprocess_ref

    if action == 'scan':
      print('ScanMain.py - running')

      session['runningProcess'] = 'ScanMain.py'
      subprocess_ref = subprocess.Popen(['python3', '/home/slc/Jetson-StructuredLight/Backend/ScanMain.py', cam_width, cam_height, graycodeDir])
      print('ScanMain.py - done')
      session['runningProcess'] = None
  
      flash("Scripts have been executed!")
    elif action == 'calib':
      print('CalibrationMain.py - running')

      session['runningProcess'] = 'CalibrationMain.py'
      subprocess_ref = subprocess.Popen(['python3', '/home/slc/Jetson-StructuredLight/Backend/CalibrationMain.py'])
      print('CalibrationMain.py - done')
      session['runningProcess'] = None
  
      flash("Scripts have been executed!")
    else:
      print("Action is not scan or calib")
  
  return redirect(url_for('index'))

@current_app.route('/cancelButtonPress', methods=['POST'])
def cancelButtonPress():
  global subprocess_ref

  if session['runningProcess']:
    subprocess_ref.terminate()
    subprocess_ref = None
    flash("Script has been terminated.")
  else:
    flash("No script is currently running.")
  return redirect(url_for('index'))

# Deprecated
@current_app.route('/model')
def serveModel():
  return send_file("/home/slc/Jetson-StructuredLight/Frontend/slcapp/static/bottlePC.ply", mimetype="application/ply")