#include <ros/ros.h>
#include "tkdnn_ros/tkDNNobjectDetector.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "tkdnn_ros");
  ros::NodeHandle nodeHandle("~");
  tkDNN_ros::tkDNNObjectDetector tkDNNObjectDetector(nodeHandle);

  ros::spin();
  return 0;
}
