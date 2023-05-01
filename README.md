# FacialRecDrone
Tello Drone Code that implements facial recognition and hand gestures to give commands

The Drone responds to three commands, an open palm hand for land; two open hands for backwards flight; and a fist will command 
the drone to spin 360 clockwise. 

NOTE: The code needs the proper libraries to be installed and added to project file to run; mediapipe handles hand gestures and facial_recognition was
used for recognizing faces; however the Drone will only respond to commands given by the user who has trained his/her face in the library. This must be 
done individually for user who seeks  command priority. 
