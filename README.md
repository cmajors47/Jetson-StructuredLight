# Jetson Nano Structured Light
### Welcome to the Jetson Nano Structured Light Capstone github! 

This project will allow you to make your Jetson Nano into a relatively inexpensive Structured Light Camera. A Structured Light Camera is essentially a 3D scanner that uses Graycode projections and a camera to perform triangulation to create a 3D recreation of an object from 2D images!

## Getting Started
* Download this repository from https://github.com/cmajors47/Jetson-StructuredLight using git clone. You may need to first do "sudo apt-get install git" in your terminal

* This library contains all of the code necessary for running a calibration and a scan of your specific settup


## Install Requirements

The first requirment is the use of the Jetson Nano operating system. The instructions on how to flash an SD card for the Jetson Nano can be found at https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

Once you get the Jetson operating system up and running, it is highly recommended to give the Jetson internet access of some sort, whether by sharing internet from another computer via ethernet or by using a cheap USB Wi-Fi module as we have for this project.

Open the terminal from the Jetson Nano home page and enter the lines below (you may copy and paste them)

1. git clone https://github.com/cmajors47/Jetson-StructuredLight
2. sudo apt-get update
3. sudo apt-get install python3-pip
4. pip3 install scikit-build
5. pip3 install --upgrade pip
6. pip3 install --prefer-binary opencv-contrib-python
7. sudo apt-get install nano
8. sudo apt-get install v4l-utils
9. pip3 install pytest-shutil
10. pip3 install open3d
11. pip3 install Flask
12. pip3 install gunicorn

Next we will edit the .bashrc file, this file contains startup information for the Jetson and is necessary for a few aspects of the project.

Type "nano .bashrc" into the command line to enter into the file. You can now use the scroll wheel to scroll to the very bottom of the file. At the very bottom, create a new line and paste "export OPENBLAS_CORETYPE=ARMV8" on the new line. If you want to use the Jetson Nano from SSH, you will also want to add the line "export DISPLAY=:0", if you are not using SSH or you do not know what SSH is, you should skip the second line, but make sure you add the first.

We will need to create a few more lines in the .bashrc file to enable the GUI on startup. Add these lines under everything else:

BEFORE ADDING THESE THREE LINES SEE THE INTIAL MESSAGE UNDER "CONNECTION TO THE JETSON FROM A WINDOWS PC". THESE CAN BE SAFELY SKIPPED FOR THE TIME BEING AND YOU CAN JUMP TO THE LINE AFTER

1. export FLASK_APP=slcapp
2. export FLASK_ENV=development
3. /home/slc/Jetson-StructuredLight/Frontend/start_gunicorn.sh

Now press control+S to save the file, and control + X to exit the file. To make sure these changes are applied, type "sudo reboot" into the terminal and enter your password when prompted, this will restart the Jetson Nano and apply our changes in the .bashrc file.

### Inputs

The following are the necessary inputs that must be known and entered into the GUI:
1. Height in pixels of the projector
2. Width in pixels of the projector
3. Height in pixels of the camera
4. Width in pixels of the camera
5. Number of vertices of the chessboard on the verticle plane
6. Number of vertices of the chessboard on the horizontal plane
7. Size of chessboard squares in milimeters
8. Step size of the graycode, this should not have to change unless you are testing different orientations of patterns.


## Connection To The Jetson From a Windows PC

This step is not currently necessary and you can skip to the "Using The Structured Light Camera" section. There have been many updates to the Backend in the past 2 weeks and there has not been time to update the Frontend to compensate for this. An update will be pushed to this github to update this at some point in the future and this message will be removed.

The first step once you obtain all of this information is to connect a laptop or computer to the Jetson Nano microcontroller via Ethernet cable.

This can be done by setting a static IP to the Jetson Nano and then communicating with it by setting your main PC's ethernet IP to the same one you set the Jetson to.

1. Using Ubuntu, click on the settings cog on the task bar to open the system settings
2. Once in the system settings access the Network icon
3. Select the wired connection you will be using, on the Jetson nano there should only be one
4. Click options at the bottom right corner
5. Select the IPv4 Settings tab
6. Change the method to "Shared to other computers"
7. Click the add button and create an address that you would like to use. I used 192.168.1.10
8. Select save

Now we have set a static IP for the Jetson Nano. We will move on to connecting to the Jetson using a Windows PC. 

1. Open up control pannel for your windows PC
2. Select "Network and Internet"
3. Select "Network and Sharing Center"
4. Now connect the ethernet from the Jetson Nano to the Windows PC
5. We should now see an option pop up in the Network and Sharing Center showing the ethernet connection.
6. Select the blue ethernet button next to "Connections:"
7. Select "Properties" at the bottom
8. Selection Internet Protocal Version 4 and select properties
9. Select the dot that says "Use the following IP address"
10. Set the IP as the same as the Jetson, except change the last value which in my case was "10" for the Jetson. The IPs cannot be identical but must share the first 3 entries.

The PC and the Jetson should now be able to communicate with each other.

Once you have done this, you can follow the README in the Frontend Folder to open the GUI and perform a scan.


## Using The Structured Light Camera

Open the terminal again and run this command to get to the right location:

cd Jetson-StructuredLight/Backend

Place a board with checkerboard pattern (default is 6 vertical vertices and 8 horizontal vertices, this can be changed in the code but recommended to stay the same) in front of the projector projection, and place the camera to the left or right of the projector, pointing at the center of the pattern.

Run this: python3 CalibrationMain.py

This will take a little while to run, and there will be 4 times that it will pause. You must change the angle and or distance of the pattern from the camera while it is still getting the projection on it and the camera can still see it. DO NOT MOVE THE CAMERA OR PROJECTOR.

Once you have changed the angle and or distance of the pattern click any key on the keyboard to continue.

Once this has completed, you will know when you click a button on the keyboard during a pause and nothing seems to happen, you will have to wait a few seconds for it to complete some calibration calculations.

Without moving the camera or projector, place an object in front of the camera and projector, preferably near where you placed the checkerboard. Next run : python3 ScanMain.py

ScanMain.py will then display images and take pictures, it will also generate the point cloud in a .ply file, and a mesh in a .stl file. ScanMain will take a few minutes to complete, you will have to give it some time.

This will be saved in the Backend folder and you can open them in a free aplication like meshlabs. 

