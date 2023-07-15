#include "model.h"
#include <iostream>
#include "process.h"
#include "BYTETracker.h"
#include <opencv2/opencv.hpp>
#include "io.h"
#include "cuda_engine.h"

using namespace std;

void Inference(CUDA_ENGINE *cudaEngine, const string &video_path) {
    cv::Mat image;
//    cv::VideoCapture videoCapture("rtsp://admin:123456@192.168.31.31/stream0");
    cv::VideoCapture videoCapture(video_path);
    int num_frames = 0;
    int total_ms = 0;
    int fps = int(videoCapture.get(CAP_PROP_FPS));
    long nFrame = static_cast<long>(videoCapture.get(CAP_PROP_FRAME_COUNT));
    BYTETracker tracker(fps, 5, nFrame);

    STrack::reset_frame_id(); // 这里进行reset标签序号

    vector<STrack> bt_outputs;  // 追踪结果集
    while (char(cv::waitKey(10) != 27) && videoCapture.isOpened() && videoCapture.read(image)) {
        auto t_beg = std::chrono::high_resolution_clock::now();
        bt_outputs.clear(); // 每次清空
        if (image.empty()) continue;
        num_frames++;
        float scale = 1; // 后面推理会修改
        std::vector<Detection> res;
        cudaEngine->do_interface(res, image, scale);
        for (int i = 0; i < res.size(); ++i) {
            getRect(res[i].rect, scale);
        }
        bt_outputs = tracker.update(res);
        cv::line(image, point_begin, point_end, Scalar_<float>(0, 0, 255), 1, LINE_AA, 0);
        for (int i = 0; i < bt_outputs.size(); i++) {
            vector<float> tlwh = bt_outputs[i].tlwh;
//            cout << "tracker:"<<num_frames<<"/"<<nFrame;
//            for (int j = 0; j < 4; ++j) {
//                cout << "\t"<<tlwh[j];
//            }
//            cout << endl;
            if (tlwh[2]* tlwh[3] > 20 /*&& !vertical*/) {
                Scalar s = tracker.get_color(bt_outputs[i].track_id);
                int label_id = bt_outputs[i].label_id;
                std::string label = cudaEngine->labels[label_id];
                std::string text = format("%d", bt_outputs[i].track_id) + label+ to_string(bt_outputs[i].score);
                cv::Rect rect;
                rect.x = bt_outputs[i].tlwh[0];
                rect.y = bt_outputs[i].tlwh[1];
                rect.width = bt_outputs[i].tlwh[2];
                rect.height = bt_outputs[i].tlwh[3];
                drawBboxMsg(image, rect, text);
            }
        }
        cv::imshow("Inference", image);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
        total_ms += int(total_inf);
    }
}


int main() {
    CUDA_ENGINE *cudaEngine = new CUDA_ENGINE();
    _finddata_t fileinfo;
    std::intptr_t handle = _findfirst("videos/*.mp4", &fileinfo);
    if (handle == -1) {
        cout << "Not Found Videos" << endl;
        return -3;
    }
    do {
        if (fileinfo.attrib != _A_SUBDIR) {
            Inference(cudaEngine, "videos/" + std::string(fileinfo.name));
        }
    } while (!_findnext(handle, &fileinfo));
    _findclose(handle);
    cv::destroyAllWindows();
    delete (cudaEngine);
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

