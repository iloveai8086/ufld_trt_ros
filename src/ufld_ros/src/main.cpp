#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "utils.hpp"
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>


#define USE_FP16  
#define DEVICE 0  
#define BATCH_SIZE 1
static const int INPUT_C = 3;
static const int INPUT_H = 288;         
static const int INPUT_W = 800;
static const int OUTPUT_C = 201;  
static const int OUTPUT_H = 18;
static const int OUTPUT_W = 4;
static const int OUTPUT_SIZE = OUTPUT_C * OUTPUT_H * OUTPUT_W;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


//static float data[1 * 3 * INPUT_H * INPUT_W];
//static float prob[1 * OUTPUT_SIZE];
static float data[BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W];
static float prob[BATCH_SIZE * OUTPUT_SIZE];

std::vector<float> processImage(cv::Mat &img);

void argmax(float *x, float *y, int rows, int cols, int chan);

void softmax_mul(float *x, float *y, int rows, int cols, int chan);

class UFLD_ROS {
public:
    UFLD_ROS(ros::NodeHandle nh) {
        cudaSetDevice(DEVICE);
        image_transport::ImageTransport it(nh);
        m_image_pub = it.advertise("/lane_det", 1);
        m_sub = it.subscribe("/usb_cam2/image_raw", 1, &UFLD_ROS::imageCallback, this);
        // TODO
        
        std::string engine_name;  
        nh.param("model_name", engine_name, std::string(
                "./lane_det.engine"));
        nh.param("batch_size", m_batch_size, 1);
        ROS_INFO("\033[1;32m----> engine_name: %s\033[0m", engine_name.c_str());
        if (!readFile(engine_name))
            return;
        initEngine();  
    }

    bool readFile(std::string engineFileName) {
        file = std::ifstream(engineFileName, std::ios::binary); 
        if (!file.good()) {
            std::cerr << "read " << engineFileName << " error!" << std::endl;
            return false;
        }
        return true;
    }

    void initEngine() {
        char *trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        printf("m_batch_size: %d --- data_size: %d!!!\n", m_batch_size, m_batch_size * 3 * INPUT_H * INPUT_W);
        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        assert(engine->getNbBindings() == 2);
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        m_inputIndex = inputIndex;
        m_outputIndex = outputIndex;
        CHECK(cudaMalloc(&buffers[inputIndex], m_batch_size * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], m_batch_size * OUTPUT_SIZE * sizeof(float)));
        CHECK(cudaStreamCreate(&stream));
        double dWidth = 800.0;
        double dHeight = 288.0; 
        std::cout << "Resolution of the video : " << dWidth << " x " << dHeight << std::endl;

    }

    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output,
                     int batchSize) {
        CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                              stream));
        cudaStreamSynchronize(stream);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        if (cv_ptr->image.empty()) return;
        int fcount = 0;
        int vis_h = 720;
        int vis_w = 1280;
        int col_sample_w = 4;  

        cv::Mat vis;  
        fcount++;
        for (int b = 0; b < fcount; b++)  
        {
            cv::Mat img = cv_ptr->image;
            if (img.empty()) continue;
            cv::resize(img, vis, cv::Size(vis_w, vis_h));

            std::vector<float> result(INPUT_C * INPUT_W * INPUT_H);
            result = processImage(img);
            memcpy(data, &result[0], INPUT_C * INPUT_W * INPUT_H * sizeof(float));  
        }

        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, m_batch_size);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time is " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us" << std::endl;


        float max_ind[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        float prob_reverse[BATCH_SIZE * OUTPUT_SIZE];
        /* do out_j = out_j[:, ::-1, :] in python list*/
        float expect[BATCH_SIZE * OUTPUT_H * OUTPUT_W];
        for (int k = 0, wh = OUTPUT_W * OUTPUT_H; k < OUTPUT_C; k++) {
            for (int j = 0; j < OUTPUT_H; j++) {
                for (int l = 0; l < OUTPUT_W; l++) {
                    prob_reverse[k * wh + (OUTPUT_H - 1 - j) * OUTPUT_W + l] =
                            prob[k * wh + j * OUTPUT_W + l];
                }
            }
        }

        argmax(prob_reverse, max_ind, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        /* calculate softmax and Expect */
        softmax_mul(prob_reverse, expect, OUTPUT_H, OUTPUT_W, OUTPUT_C);
        for (int k = 0; k < OUTPUT_H; k++) {
            for (int j = 0; j < OUTPUT_W; j++) {
                max_ind[k * OUTPUT_W + j] == 200 ? expect[k * OUTPUT_W + j] = 0 :  
                        expect[k * OUTPUT_W + j] = expect[k * OUTPUT_W + j];
            }
        }
        std::vector<int> i_ind;
        for (int k = 0; k < OUTPUT_W; k++) {
            int ii = 0;
            for (int g = 0; g < OUTPUT_H; g++) {
                if (expect[g * OUTPUT_W + k] != 0)
                    ii++;
            }
            if (ii > 2) {
                i_ind.push_back(k);
            }
        }

        for (int k = 0; k < OUTPUT_H; k++) {
            for (int ll = 0; ll < i_ind.size(); ll++) {
                if (expect[OUTPUT_W * k + i_ind[ll]] > 0) {
                    cv::Point pp =
                            {int(expect[OUTPUT_W * k + i_ind[ll]] * col_sample_w * vis_w / INPUT_W) - 1,
                             int(vis_h * tusimple_row_anchor[OUTPUT_H - 1 - k] / INPUT_H) - 1};
                    cv::circle(vis, pp, 8, CV_RGB(0, 255, 0), 2);
                }
            }
        }

        sensor_msgs::Image img_msg;
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, vis);
        img_bridge.toImageMsg(img_msg);
        m_image_pub.publish(img_msg);

    }

    void clearMemory() {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[m_inputIndex]));
        CHECK(cudaFree(buffers[m_outputIndex]));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }


private:
    std::ifstream file;
    cv_bridge::CvImage img_bridge;
    image_transport::Publisher m_image_pub;
    image_transport::Subscriber m_sub;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    double m_nms_thresh, m_conf_thresh;
    int m_batch_size;
    cudaStream_t stream;
    int m_inputIndex, m_outputIndex;
    void *buffers[2];
    std::vector<int> tusimple_row_anchor
            {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
};



std::vector<float> processImage(cv::Mat &img) 
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat img_float;

    resized.convertTo(img_float, CV_32FC3, 1. / 255.);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(INPUT_C);
    cv::split(img_float, input_channels);

    // normalize
    std::vector<float> result(INPUT_H * INPUT_W * INPUT_C);
    auto data = result.data();
    int channelLength = INPUT_H * INPUT_W;
    static float mean[] = {0.485, 0.456, 0.406};
    static float std[] = {0.229, 0.224, 0.225};
    for (int i = 0; i < INPUT_C; ++i) {
        cv::Mat normed_channel = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, normed_channel.data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result; 
}

void softmax_mul(float *x, float *y, int rows, int cols, int chan) {
    for (int i = 0, wh = rows * cols; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0;
            float expect = 0.0;
            for (int k = 0; k < chan - 1; k++) {
                x[k * wh + i * cols + j] = exp(x[k * wh + i * cols + j]);
                sum += x[k * wh + i * cols + j];
            }
            for (int k = 0; k < chan - 1; k++) {
                x[k * wh + i * cols + j] /= sum;
            }
            for (int k = 0; k < chan - 1; k++) {
                x[k * wh + i * cols + j] = x[k * wh + i * cols + j] * (k + 1);
                expect += x[k * wh + i * cols + j];
            }
            y[i * cols + j] = expect;
        }
    }
}


void argmax(float *x, float *y, int rows, int cols, int chan) {
    for (int i = 0, wh = rows * cols; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int max = -10000000;
            int max_ind = -1;
            for (int k = 0; k < chan; k++) {
                if (x[k * wh + i * cols + j] > max) {
                    max = x[k * wh + i * cols + j];
                    max_ind = k;
                }
            }
            y[i * cols + j] = max_ind;
        }
    }
}

int main(int argc, char **argv) {

    ros::init(argc, argv, "ufld_ros");
    ros::NodeHandle nh("~");
    UFLD_ROS ufld_ros = UFLD_ROS(nh);
    ros::spin();
    ufld_ros.clearMemory();



    return 0;
}


