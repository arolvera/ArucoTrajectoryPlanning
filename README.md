# Aruco markers for pose estimation and augmented reality

This repository uses openCV for the pose and kinematic estimation of ARUCO markers ID's 0, 1 and 2
Each ID represents a seperate physical object with different handling requirements. Certain objects 
that are more fragile will have constraints on their max velocity When a tag is detected, its shape 
is displayed using AR and the velocity of the object is detected, visual cues are provided to the 
user such as a colorscale and warning sign overlaid using a homography when the object reaches a 
critical speed. More details can be found in the PDF. The functionality is implemented in the 
ArucoPoseEstimation.py script. Note that different cameras will have different parameters, thus, 
these patrameters need to be obtained using chessboard images and the calibration script provided. 
First run main_chessboard.py using your webcam and the provided chesboard printout. This will save 
the images needed for calibration. Next run main_calib.py to get your cameras parameters.  