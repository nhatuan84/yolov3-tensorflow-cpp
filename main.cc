#include <fstream>
#include <utility>
#include <vector>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>


#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::ops::Softmax;

#define printTensor(T, d) \
    std::cout<< (T).tensor<float, (d)>() << std::endl;

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#define YOLOV3_SIZE     416
#define IMG_CHANNELS    3


float bboxThreshold = 0.4; // BBox threshold
float nmsThreshold = 0.4; // Non-maximum suppression threshold
std::vector<string> classes;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

cv::Mat resizeKeepAspectRatio(const cv::Mat &input, int width, int height)
{
    cv::Mat output;

    double h1 = width * (input.rows/(double)input.cols);
    double w2 = height * (input.cols/(double)input.rows);
    if( h1 <= height) {
        cv::resize( input, output, cv::Size(width, h1));
    } else {
        cv::resize( input, output, cv::Size(w2, height));
    }

    int top = (height - output.rows) / 2;
    int down = (height - output.rows + 1) / 2;
    int left = (width - output.cols) / 2;
    int right = (width - output.cols + 1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(128,128,128) );

    return output;
}

Status readTensorFromMat(const cv::Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;
    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    cv::Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3, 1.f);
    
    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto noOp = Identity(root.WithOpName("noOp"), outTensor);


    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::vector<Tensor> outTensors;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"noOp"}, {}, &outTensors));
    
    outTensor = outTensors.at(0);
    return Status::OK();
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 2);
    
    //Get the label for the class name and its confidence
    string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), 
                  cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            //// Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (data[4] > bboxThreshold)
            {
                int x0 = (int)(data[0]);
                int y0 = (int)(data[1]);
                int x1 = (int)(data[2]);
                int y1 = (int)(data[3]);
                
                //recover bbox according to input size
                int current_size = YOLOV3_SIZE;
                int rows = frame.rows;
                int cols = frame.cols;
                float final_ratio = std::min((float)current_size/cols, (float)current_size/rows);
                int padx = 0.5f * (current_size - final_ratio * cols);
                int pady = 0.5f * (current_size - final_ratio * rows);
                
                x0 = (x0 - padx) / final_ratio;
                y0 = (y0 - pady) / final_ratio;
                x1 = (x1 - padx) / final_ratio;
                y1 = (y1 - pady) / final_ratio;
              
                int left = x0;
                int top = y0;
                int width = x1 - x0;
                int height = y1 - y0;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, bboxThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, char* argv[]) {

  string dataset = "dataset/";
  string graph = "model/frozen_model.pb";
  std::vector<string> files;
  string input_layer = "inputs"; //input ops
  string final_out = "output_boxes"; //output ops
  string root_dir = "";
  
  string classesFile = "coco.names";
  std::ifstream ifs(classesFile.c_str());
  string line;
  while (getline(ifs, line)) classes.push_back(line);

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n";
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  cv::VideoCapture cap;
  if(!cap.open(0)) {
        return 0;
  }
  
  for(;;) {

    cv::Mat srcImage, rgbImage;
    cap >> srcImage;
    if(srcImage.empty()){
        break;
    }
    cv::cvtColor(srcImage, rgbImage, CV_BGR2RGB);
    cv::Mat padImage = resizeKeepAspectRatio(rgbImage, YOLOV3_SIZE, YOLOV3_SIZE);

    Tensor resized_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, YOLOV3_SIZE, YOLOV3_SIZE, IMG_CHANNELS}));
    Status read_tensor_status = readTensorFromMat(padImage, resized_tensor);
    if (!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return -1;
    }
      
    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     {final_out}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }
    //std::cout << outputs[0].shape() << "\n";
    float *p = outputs[0].flat<float>().data();
    cv::Mat result(outputs[0].dim_size(1), outputs[0].dim_size(2), CV_32FC(1), p);
    std::vector<cv::Mat> outs;
    outs.push_back (result);
    
    postprocess(rgbImage, outs);
    
    cv::cvtColor(rgbImage, srcImage , CV_RGB2BGR);   
    cv::imshow( "Yolov3", srcImage );
    if( cv::waitKey(10) == 27 ) break;
    
  }
  return 0;
}
