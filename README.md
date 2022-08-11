# Lane Detection in TensorRT and ROS


##ENV
```
Jetson:
jetpack with tensorrt7.x,if the version of tensorrt is higher than 7.x,some of 
the API in build_engine in this code should be changed.
X86:
CUDA10.2-11.2
TensorRT7
ROS-melodic
opencv3/4
```

##Before run
This is just an inference code, to generating the engine file,you can
go to this page <https://github.com/YZY-stack/Ultra_Fast_Lane_Detection_TensorRT/tree/master/UFLD_C%2B%2B>



## how to build and run
```
cd workspcae
catkin_make
source devel/setup.bash
roslaunch ufld_ros ufld_ros.launch

open another bash terminal
roscore

open another bash terminal
if you want to use the rosbag with copressed_raw you should run:
rosrun image_transport republish compressed in:=/gmsl/front/image_raw raw out:=/usb_cam2/image_raw
and then play the rosbag 

in rviz choose the lane_det topic for visualizing the results
```



