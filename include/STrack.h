#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

enum TrackState {
    New = 0, Tracked, Lost, Removed
};

class STrack {
public:
    explicit STrack(vector<float> tlwh_, float score=0, int _label_id=0);

    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);

    void static multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);

    void static_tlwh();

    void static_tlbr();

    static vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);

    vector<float> to_xyah() const;

    void mark_lost();

    void mark_removed();

    static int _count;

    static int next_id();

    int end_frame() const;

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);

    void re_activate(STrack &new_track, int frame_id, bool new_id = false);

    void update(STrack &new_track, int frame_id);

    static void reset_frame_id();

public:
    bool is_activated;
    int track_id; // 追踪ID
    int label_id; // 标签ID
    int state;

    vector<float> _tlwh;
    vector<float> tlwh;
    vector<float> tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};