# Jetson Nano Structured Light

* Download this repository from https://github.com/cmajors47/Jetson-StructuredLight using git clone. You may need to first do "sudo apt-get install git" in your terminal

* This library contains all of the code necessary for running a calibration and a scan of your specific settup

## Install Requirements
### You May copy and paste the following lines into your terminal:

1. sudo apt-get install opencv_contrib_python
2. sudo apt-get install pip
3. pip3 install glob
4. pip3 install 


























The following are the necessary inputs that must be known and entered into the GUI:
1. Height in pixels of the projector
2. Width in pixels of the projector
3. Height in pixels of the camera
4. Width in pixels of the camera
5. Number of vertices of the chessboard on the verticle plane
6. Number of vertices of the chessboard on the horizontal plane
7. Size of chessboard squares in milimeters
8. Step size of the graycode, this should not have to change unless you are testing different orientations of patterns.























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
