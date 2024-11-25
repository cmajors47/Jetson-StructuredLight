# Structured-Light-Camera
The Structured Light Camera will scan objects and output the 3D model as a CAD compatible file. The system will use an integrated circuit and be a compact device with modular design. This repository will house the code that will operate the system on the frontend.

To set up the dependencies, a Python environment must be set up. The easiest way to do this is to set up a venv (NOTE: Python must be installed on the machine). The command line to do this is:
```python -m venv slcenv```

To start the web application for debugging, use the following command in the directory that houses the slc-flask and slcapp folders.
```powershell
python -m flask --app slcapp run --debug
```

## Prototype Deployment
Run ifconfig to find the inet value (this is the IP address)

cd into the Structured-Light-Camera-Frontend folder and run the following command to start the debug server
```python -m flask --app slcapp run --host=[IPaddress] --port=[portno]```

On Jetson this command will be:
```export FLASK_APP=slcapp```
```export FLASK_ENV=development```
```python3 -m flask run```

Now the Flask web app should be viewable from the connected device by typing the following URL into a web browser:
```
http://[IPaddress]:[portno]
```

## Deployment (maybe save this for later)
```
server {
  listen 80;
  server_name example.org;
  access_log  /var/log/nginx/example.log;

  location / {
      proxy_pass http://127.0.0.1:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
}
```

```gunicorn -b 127.0.0.1:8000 "slcapp:create_app"```
