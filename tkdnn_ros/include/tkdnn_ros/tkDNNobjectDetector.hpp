#pragma once

// c++
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ROS
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

// OpenCv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// darknet_ros_msgs
#include <tkdnn_ros_msgs/BoundingBox.h>
#include <tkdnn_ros_msgs/BoundingBoxes.h>
#include <tkdnn_ros_msgs/CheckForObjectsAction.h>
#include <tkdnn_ros_msgs/ObjectCount.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
#include <cudnn.h>

#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

namespace tkDNN_ros {

//! Bounding box of the detected object.
typedef struct {
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;

class tkDNNObjectDetector {
 public:
  /*!
   * Constructor.
   */
  explicit tkDNNObjectDetector(ros::NodeHandle nh);

  /*!
   * Destructor.
   */
  ~tkDNNObjectDetector();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Initialize the ROS connections.
   */
  void init();

  /*!
   * Callback of camera.
   * @param[in] msg image pointer.
   */
  void cameraCallback(const sensor_msgs::ImageConstPtr& msg);

  /*!
   * Publishes the detection image.
   * @return true if successful.
   */
  bool publishDetectionImage(const std::vector<cv::Mat>& detectionImage);

  //! ROS node handle.
  ros::NodeHandle nodeHandle_;

  tk::dnn::Yolo3Detection yolo;
  tk::dnn::CenternetDetection cnet;
  tk::dnn::MobilenetDetection mbnet; 

  tk::dnn::DetectionNN *detNN;  

  std::vector<cv::Mat> batch_frame;
  std::vector<cv::Mat> batch_dnn_input; 

  bool running;
  bool viewImage_;
  bool enableConsoleOutput_;

  int waitKeyDelay_;
  int n_batch;

  //! Class labels.
  int numClasses_;
  std::vector<std::string> classLabels_;

  //! Advertise and subscribe to image topics.
  image_transport::ImageTransport imageTransport_;

  //! ROS subscriber and publisher.
  image_transport::Subscriber imageSubscriber_;
  ros::Publisher boundingBoxesPublisher_;
  ros::Publisher objectPublisher_;
  ros::Publisher detectionImagePublisher_;
  int8_t ObjectResultCount;
  float fps;

  //! Detected objects.
  tkdnn_ros_msgs::BoundingBoxes boundingBoxesResults_;
  
  //! Camera related parameters.
  int frameWidth_;
  int frameHeight_;

  // tkDNN running on thread.
  std::thread tkDNNThread_;

  RosBox_* roiBoxes_;
  std::vector<std::vector<RosBox_> > rosBoxes_; 
  std::vector<int> rosBoxCounter_; 

  std_msgs::Header imageHeader_;

  cv::Mat camImageCopy_,camImageCopy_tkdnn;
  boost::shared_mutex mutexImageCallback_;

  bool imageStatus_ = false;
  boost::shared_mutex mutexImageStatus_;

  bool isNodeRunning_ = true;
  boost::shared_mutex mutexNodeStatus_;

  cv::Mat& fetchInThread();

  void* detectInThread(std::vector<cv::Mat>& frames);
  
  void tkDNN();
  
  bool getImageStatus(void);

  bool isNodeRunning(void);

  void* publishInThread(std::vector<cv::Mat>& frames);
};

} /* namespace tkdnn_ros*/
