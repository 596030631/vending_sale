#include "model.h"
#include <iostream>
#include "process.h"
#include "BYTETracker.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "io.h"
#include "cuda_engine.h"

using namespace std;

//static const int INPUT_W = 640;
//static const int INPUT_H = 640;
//const char *INPUT_BLOB_NAME = "input_0";
//const char *OUTPUT_BLOB_NAME = "output_0";



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


void Inference(CUDA_ENGINE *cudaEngine, const string& video_path) {

    cv::Mat image;
//    cv::VideoCapture videoCapture("rtsp://admin:123456@192.168.31.31/stream0");
    cv::VideoCapture videoCapture(video_path);

    int num_frames = 0;
    int total_ms = 0;
    int img_w = int(videoCapture.get(CAP_PROP_FRAME_WIDTH));
    int img_h = int(videoCapture.get(CAP_PROP_FRAME_HEIGHT));
    int fps = int(videoCapture.get(CAP_PROP_FPS));
    long nFrame = static_cast<long>(videoCapture.get(CAP_PROP_FRAME_COUNT));
    BYTETracker tracker(fps, 30);
    cout << "Total frames: " << nFrame << endl;

    while (char(cv::waitKey(1) != 27) && videoCapture.isOpened() && videoCapture.read(image)) {
        auto t_beg = std::chrono::high_resolution_clock::now();
        if (image.empty()) continue;

        num_frames++;

        float scale = 1; // 后面推理会修改

        std::vector<Detection> res;

        cudaEngine->do_interface(res, image, scale);

        for (int i = 0; i < res.size(); ++i) {
            getRect(image, &res[i].rect, scale);
        }
        vector<STrack> output_stracks = tracker.update(res);

        cv::line(image, point_begin, point_end, Scalar_<float>(0, 0, 255), 1, LINE_AA, 0);


        for (int i = 0; i < output_stracks.size(); i++) {
            vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20 /*&& !vertical*/) {
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(image, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5),
                        0, 0.6, s, 1, LINE_AA);
//                    rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);

//                float x = (2.0f * tlwh[0] + tlwh[2]) / 2.0f;
//                float y = (2.0f * tlwh[1] + tlwh[3]) / 2.0f;
//
//                cout << "id=" << output_stracks[i].track_id << "\ttlwh=" << tlwh[0] << " " << tlwh[1] << " " << tlwh[2]
//                     << " " << tlwh[3] << "center=(" << x << "," << y << ") tracked_len="
//                     << output_stracks[i].tracklet_len << endl;




//                cout << "out_size" <<  << endl;
            }
        }

        drawBbox(image, res, scale, cudaEngine->labels);


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



}


int main(int argc) {

    CUDA_ENGINE *cudaEngine = new CUDA_ENGINE();

    _finddata_t fileinfo;
    string ext = ".mp4";
    std::intptr_t handle = _findfirst("./*.mp4", &fileinfo);
    if (handle == -1) {
        cout << "本地文件查找失败" << endl;
        return -3;
    }

    do {
//        string attr;
//        if (fileinfo.attrib != _A_SUBDIR) {
        cout << "file_name=" << fileinfo.name << endl;
        Inference(cudaEngine, fileinfo.name);
//        }
//
//
    } while (!_findnext(handle, &fileinfo));
//
    _findclose(handle);

    cv::destroyAllWindows();
    delete(cudaEngine);

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

