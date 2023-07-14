#include "model.h"
#include "utils.h"
#include <iostream>
#include "process.h"
#include "BYTETracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;

Logger gLogger;
//static const int INPUT_W = 640;
//static const int INPUT_H = 640;
//const char *INPUT_BLOB_NAME = "input_0";
//const char *OUTPUT_BLOB_NAME = "output_0";

void serializeEngine(const int &kBatchSize, std::string &wts_name, std::string &engine_name, std::string &sub_type) {

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n") {
        serialized_engine = buildEngineYolov8n(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    } else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    } else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    } else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    } else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}


//Mat static_resize(Mat &img) {
//    float r = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
//    // r = std::min(r, 1.0f);
//    int unpad_w = r * img.cols;
//    int unpad_h = r * img.rows;
//    Mat re(unpad_h, unpad_w, CV_8UC3);
//    resize(img, re, re.size());
//    Mat out(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));
//    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));
//    return out;
//}


void Inference(std::string &engine_name, std::string &class_file) {
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;

    readEngineFile(engine_name, runtime, engine, context);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    float *device_buffers[2];
    uint8_t *image_device = nullptr;
    auto *output_buffer_host = new float[kBatchSize * kOutputSize];
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaMalloc((void **) &image_device, kMaxInputImageSize * 3);
    cudaMalloc((void **) &device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void **) &device_buffers[1], kBatchSize * kOutputSize * sizeof(float));

    std::map<int, std::string> labels;
    readClassFile(class_file, labels);

    cv::Mat image;
//    cv::VideoCapture videoCapture("rtsp://admin:123456@192.168.31.31/stream0");
    cv::VideoCapture videoCapture("test.mp4");

    int num_frames = 0;
    int total_ms = 0;
    int img_w = int(videoCapture.get(CAP_PROP_FRAME_WIDTH));
    int img_h = int(videoCapture.get(CAP_PROP_FRAME_HEIGHT));
    int fps = int(videoCapture.get(CAP_PROP_FPS));
    long nFrame = static_cast<long>(videoCapture.get(CAP_PROP_FRAME_COUNT));
    BYTETracker tracker(fps, 30);
    cout << "Total frames: " << nFrame << endl;

    struct Goods {
        int id = 0;
        float rect_begin[4]{0, 0, 0, 0};
        float rect_end[4]{0, 0, 0, 0};
    };

    vector<Goods> goods;


    while (char(cv::waitKey(30) != 27)) {
        auto t_beg = std::chrono::high_resolution_clock::now();

        videoCapture >> image;

        if (image.empty()) continue;

        cout << "----------------------" << endl;

        num_frames++;

        float scale = 640.00f / float(std::min(img_w, img_h));
        int img_size = image.cols * image.rows * 3;
        cudaMemcpyAsync(image_device, image.data, img_size, cudaMemcpyHostToDevice, stream);
        preprocess(image_device, image.cols, image.rows, device_buffers[0], kInputW, kInputH, stream, scale);
        context->enqueue(kBatchSize, (void **) device_buffers, stream, nullptr);
        cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<Detection> res;
        NMS(res, output_buffer_host, kConfThresh, kNmsThresh);

        for (int i = 0; i < res.size(); ++i) {
            getRect(image, &res[i].rect, scale);
        }

        vector<STrack> output_stracks = tracker.update(res);


        for (int i = 0; i < output_stracks.size(); i++) {
            vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20 /*&& !vertical*/) {
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(image, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, s, 1, LINE_AA);
//                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
                cout << "id=" << output_stracks[i].track_id << "\ttlwh=" << tlwh[0] << " " << tlwh[1] << " " << tlwh[2]
                     << " " << tlwh[3] << endl;

                float x = (2.0f * tlwh[0] + tlwh[2]) / 2.0f;
                float y = (2.0f * tlwh[1] + tlwh[3]) / 2.0f;
                cout << "center=(" << x << "," << y << ") tracked_len="<<output_stracks[i].tracklet_len << endl;
            }
        }

        drawBbox(image, res, scale, labels);


        cv::imshow("Inference", image);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        total_ms += int(total_inf);
//        cout << "Inference time: " << int(total_inf) << " count=" << res.size() << std::endl;
//        cout << "FPS: " << num_frames * 1000 / total_ms << endl;


//            putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()),
//                    Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
//            writer.write(img);

    }

    cv::destroyAllWindows();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    delete[] output_buffer_host;
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}


int main(int argc) {
    std::string sub_type = "m";
    string model_name = "yolov8" + sub_type + "_n1";
    string wts_name = "../weights/" + model_name + ".wts";
    string engine_name = "../weights/" + model_name + "_b" + to_string(kBatchSize) + ".engine";
    string class_file = "../weights/classes1.txt";

    fstream f(engine_name.c_str());
    if (!f.good()) {
        serializeEngine(kBatchSize, wts_name, engine_name, sub_type);
        return 0;
    }
    Inference(engine_name, class_file);
    return 0;
}

/****************************************************************************************************
******************************************  dllexport  **********************************************
*****************************************************************************************************/

//extern "C" __declspec(dllexport) void startInference() {
//    std::string engine_name = "./weights/yolov8s_b1.engine";
//    std::string class_file = "./weights/classes80.txt";
//
//    Inference(engine_name, class_file);
//
//    return;
//}

