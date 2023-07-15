//
// Created by SJ on 2023/7/15.
//

#include "cuda_engine.h"

//using namespace std;

Logger gLogger;

CUDA_ENGINE::CUDA_ENGINE() : image_device(nullptr), output_buffer_host(nullptr), stream(nullptr), runtime(nullptr),
                             engine(nullptr), context(nullptr) {
    fstream f(engine_name.c_str());
    if (!f.good()) {
        serializeEngine(BATCH_SIZE, wts_name, engine_name, sub_type);
    }
    this->kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    this->output_buffer_host = new float[BATCH_SIZE * kOutputSize];

    init_infer(engine_name, class_file);
}

CUDA_ENGINE::~CUDA_ENGINE() {
    std::cout << "release cuda" << endl;
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

void CUDA_ENGINE::serializeEngine(const int &_kBatchSize, const std::string &_wts_name, const std::string &_engine_name,
                                  const std::string &_sub_type) {

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::IHostMemory *serialized_engine = nullptr;

    if (_sub_type == "n") {
        serialized_engine = buildEngineYolov8n(_kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, _wts_name);
    } else if (_sub_type == "s") {
        serialized_engine = buildEngineYolov8s(_kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, _wts_name);
    } else if (_sub_type == "m") {
        serialized_engine = buildEngineYolov8m(_kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, _wts_name);
    } else if (_sub_type == "l") {
        serialized_engine = buildEngineYolov8l(_kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, _wts_name);
    } else if (_sub_type == "x") {
        serialized_engine = buildEngineYolov8x(_kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, _wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(_engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}


void CUDA_ENGINE::init_infer(const string &_engine_name, const string &_class_file) {
    readEngineFile(_engine_name, runtime, engine, context);
    cudaStreamCreate(&stream);

    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    cudaMalloc((void **) &this->image_device, kMaxInputImageSize * 3);
    cudaMalloc((void **) &this->device_buffers[0], BATCH_SIZE * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void **) &this->device_buffers[1], BATCH_SIZE * kOutputSize * sizeof(float));

    readClassFile(_class_file, this->labels);
}

void CUDA_ENGINE::do_interface() {



}
