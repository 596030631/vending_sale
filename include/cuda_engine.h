//
// Created by SJ on 2023/7/15.
//

#ifndef YOLOV8_CUDA_ENGINE_H
#define YOLOV8_CUDA_ENGINE_H

#include "ostream"
#include "utils.h"
#include "string"
#include "config.h"

#endif //YOLOV8_CUDA_ENGINE_H


using namespace std;

class CUDA_ENGINE {

private:
    static void serializeEngine(const int &_kBatchSize, const std::string &_wts_name, const std::string &_engine_name,
                         const std::string &_sub_type);

    void init_infer(const string &_engine_name, const string &_class_file);

public:
    CUDA_ENGINE();

    ~CUDA_ENGINE();

    do_interface();

    int kOutputSize;
    std::map<int, std::string> labels;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    cudaStream_t stream;
    uint8_t *image_device;
    float *device_buffers[2];
    float *output_buffer_host;
};