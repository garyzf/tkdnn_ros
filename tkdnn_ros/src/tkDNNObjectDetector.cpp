#include "tkdnn_ros/tkDNNobjectDetector.hpp"

namespace tkDNN_ros {

char** detectionNames;

tkDNNObjectDetector::tkDNNObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh), imageTransport_(nodeHandle_), numClasses_(0), classLabels_(0), rosBoxes_(0), rosBoxCounter_(0) {
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }
  init();
}

tkDNNObjectDetector::~tkDNNObjectDetector() {
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  tkDNNThread_.join();
}

bool tkDNNObjectDetector::readParameters() {
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_opencv", viewImage_, false);

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_, std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);
  return true;
}

void tkDNNObjectDetector::init() {
  ROS_INFO("[tkDNNObjectDetector] init().");

  float thresh;

  nodeHandle_.param("yolo_model/threshold/value", thresh, (float)0.3);
  nodeHandle_.param("yolo_model/batchsize/value", n_batch, 4);

  std::string configPath;
  std::string configModel;
  std::string net_;
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolo4tiny_fp16.rt"));
  nodeHandle_.param("net_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  net_ =configPath;
  ROS_INFO("net_:%s",net_.c_str());
  // Get classes.
  ROS_INFO("numClasses:%d",numClasses_);
  detectionNames = (char**)realloc((void*)detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  running = true;
  char ntype ='y';
  switch(ntype)
  {
      case 'y':
          detNN = &yolo;
          break;
      case 'c':
          detNN = &cnet;
          break;
      case 'm':
          detNN = &mbnet;
          numClasses_++;
          break;
      default:break;
  }
  detNN->init(net_, numClasses_, n_batch, thresh);
  ROS_INFO("detNN->init(net_, numClasses_, n_batch, thresh);");

  //creat tkdnn thread
  tkDNNThread_ = std::thread(&tkDNNObjectDetector::tkDNN, this);
  ROS_INFO("tkDNNThread_ = std::thread(&tkDNNObjectDetector::tkDNN, this);");

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);

  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName, std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);

  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName, std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);

  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName, std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &tkDNNObjectDetector::cameraCallback, this);
  objectPublisher_ =
      nodeHandle_.advertise<tkdnn_ros_msgs::ObjectCount>(objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
  boundingBoxesPublisher_ =
      nodeHandle_.advertise<tkdnn_ros_msgs::BoundingBoxes>(boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
  
  ROS_INFO("tkDNNObjectDetector::init");
}

void tkDNNObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
  ROS_DEBUG("[YoloObjectDetector] image topic received.");
  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image){// image has
    {//write thread lock
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
    }

    {
      //write thread lock
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }

    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }

  return;
}


void* tkDNNObjectDetector::detectInThread(std::vector<cv::Mat>& frames) {
  tk::dnn::box dets;

  ObjectResultCount =0;
  for(int bi=0; bi<frames.size(); ++bi){
  // det 
    for (int i = 0; i < detNN->batchDetected[bi].size(); ++i) {
        dets =detNN->batchDetected[bi][i];
        float xmin = dets.x;
        float xmax = dets.x + dets.w;
        float ymin = dets.y;
        float ymax = dets.y + dets.h;
        tkdnn_ros_msgs::BoundingBox boundingBox;

        boundingBox.Class = classLabels_[dets.cl];
        boundingBox.id = dets.cl;
        boundingBox.probability = dets.prob;
        boundingBox.xmin = xmin;
        boundingBox.ymin = ymin;
        boundingBox.xmax = xmax;
        boundingBox.ymax = ymax;
        boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        ObjectResultCount ++;
      }
    }
  
  return 0;
}

void* tkDNNObjectDetector::publishInThread(std::vector<cv::Mat>& frames) {
  // Publish image.
  if (!publishDetectionImage(frames)) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  tkdnn_ros_msgs::ObjectCount msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "detection";
  msg.count = ObjectResultCount;
  objectPublisher_.publish(msg);

  if(ObjectResultCount >0){
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.fps =fps;
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
  }

  boundingBoxesResults_.bounding_boxes.clear();

  return 0;
}

cv::Mat& tkDNNObjectDetector::fetchInThread() {
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    camImageCopy_tkdnn =camImageCopy_.clone();
  }
   return camImageCopy_tkdnn;
}

bool tkDNNObjectDetector::publishDetectionImage(const std::vector<cv::Mat>& detectionImage) {
  if (detectionImagePublisher_.getNumSubscribers() < 1) return false;
  cv_bridge::CvImage cvImage;

  for(int bi=0; bi<detectionImage.size(); ++bi){
    cvImage.header.stamp = ros::Time::now();
    cvImage.header.frame_id = "detection_image";
    cvImage.encoding = sensor_msgs::image_encodings::BGR8;
    cvImage.image = detectionImage[bi].clone();
    detectionImagePublisher_.publish(*cvImage.toImageMsg());
  }

  ROS_DEBUG("Detection image has been published.");
  return true;
}

//main thread enter
void tkDNNObjectDetector::tkDNN() {
  //sleep 2000ms,this thread
  const auto wait_duration = std::chrono::milliseconds(2000);
  //if imageStatus_==false,No image

  while (!getImageStatus()) {
    printf("Waiting for image.\n");

    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  if(viewImage_)
      cv::namedWindow("detection", cv::WINDOW_NORMAL);

  cv::Mat frame;
  double mean = 0;

  while (running) {

    batch_dnn_input.clear();
    batch_frame.clear();

    for(int bi=0; bi< n_batch; ++bi){
        frame =fetchInThread();//read current video
        if(!frame.data)break;
        
        batch_frame.push_back(frame);
        // this will be resized to the net format
        batch_dnn_input.push_back(frame.clone());
    } 

    if(!frame.data)break;

    //inference
    detNN->update(batch_dnn_input, n_batch);
    detNN->draw(batch_frame);
    
    //cal fps or time consump
    mean =0;
    for(int i=0; i<detNN->stats.size(); i++){
        mean += detNN->stats[i];
    } 
    mean /= detNN->stats.size();
    fps =1000/(mean/n_batch);

    //publish
    detectInThread(batch_frame);
    publishInThread(batch_frame);
    //view image
    if(viewImage_){
        for(int bi=0; bi< n_batch; ++bi){
            cv::imshow("detection", batch_frame[bi]);
            cv::waitKey(1);
        }
    }

    if(enableConsoleOutput_ ==true){
      std::cout<<"Avg: "<<1000/fps<<" ms\t"<<fps<<" FPS\n"<<COL_END; 
    }

    if (!isNodeRunning()) {
      running = false;
    }
  }
}

bool tkDNNObjectDetector::getImageStatus(void) {
  // read thread unlock
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool tkDNNObjectDetector::isNodeRunning(void) {
  // read thread unlock
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

} /* namespace darknet_ros*/
