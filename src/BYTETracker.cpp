#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer, int _frame_count) {
    track_thresh = 0.5;
    high_thresh = 0.6;
    match_thresh = 0.8;
    frame_count = _frame_count;
    frame_id = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

BYTETracker::~BYTETracker() {
}

vector<STrack> BYTETracker::update(const vector<Detection> &objects) {

    ////////////////// Step 1: Get detections //////////////////
    this->frame_id++;
    vector<STrack> activated_stracks;
    vector<STrack> refind_stracks;
    vector<STrack> removed_stracks;
    vector<STrack> lost_stracks;
    vector<STrack> detections;
    vector<STrack> detections_low;

    vector<STrack> detections_cp;
    vector<STrack> tracked_stracks_swap;
    vector<STrack> resa, resb;
    vector<STrack> output_stracks;

    vector<STrack *> unconfirmed;
    vector<STrack *> tracked_stracks;
    vector<STrack *> strack_pool;
    vector<STrack *> r_tracked_stracks;

    if (objects.size() > 0) {
        for (int i = 0; i < objects.size(); i++) {
            vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = objects[i].rect.x;
            tlbr_[1] = objects[i].rect.y;
            tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
            tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

            float score = objects[i].prob;

            STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, objects[i].label);
            if (score >= track_thresh) {
                detections.push_back(strack);
            } else {
                detections_low.push_back(strack);
            }

        }
    }

    // Add newly detected tracklets to tracked_stracks
    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (!this->tracked_stracks[i].is_activated)
            unconfirmed.push_back(&this->tracked_stracks[i]);
        else
            tracked_stracks.push_back(&this->tracked_stracks[i]);
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
    STrack::multi_predict(strack_pool, this->kalman_filter);

    vector<vector<float> > dists;
    int dist_size = 0, dist_size_size = 0;
    dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

    vector<vector<int> > matches;
    vector<int> u_track, u_detection;
    linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack *track = strack_pool[matches[i][0]];
        STrack *det = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        detections_cp.push_back(detections[u_detection[i]]);
    }
    detections.clear();
    detections.assign(detections_low.begin(), detections_low.end());

    for (int i = 0; i < u_track.size(); i++) {
        if (strack_pool[u_track[i]]->state == TrackState::Tracked) {
            r_tracked_stracks.push_back(strack_pool[u_track[i]]);
        }
    }

    dists.clear();
    dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack *track = r_tracked_stracks[matches[i][0]];
        STrack *det = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    for (int i = 0; i < u_track.size(); i++) {
        STrack *track = r_tracked_stracks[u_track[i]];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    vector<int> u_unconfirmed;
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
        activated_stracks.push_back(*unconfirmed[matches[i][0]]);
    }

    for (int i = 0; i < u_unconfirmed.size(); i++) {
        STrack *track = unconfirmed[u_unconfirmed[i]];
        track->mark_removed();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        STrack *track = &detections[u_detection[i]];
        if (track->score < this->high_thresh)
            continue;
        track->activate(this->kalman_filter, this->frame_id);
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (int i = 0; i < this->lost_stracks.size(); i++) {
        if (this->lossIds.count(this->lost_stracks[i].track_id)) {
            continue;
        }

        if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost || this->frame_id >= this->frame_count) {
            this->lossIds.insert(this->lost_stracks[i].track_id);
//            std::cout << "lost=" << this->lost_stracks[i].track_id << std::endl;
            this->lost_stracks[i].mark_removed();
            removed_stracks.push_back(this->lost_stracks[i]);

//            static float x1, y1, x2, y2;
//            x1 = 0;
//            y1 = -320;
//            x2 = 1280;
//            y2 = -400;

            // 用下角的点，判断商品拿取,右开门肯定是左下，左开门肯定是右下，取两者最小值能最大限度地判断商品出界
            float xr = this->lost_stracks[i].tlwh[0] + this->lost_stracks[i].tlwh[2];
            float xl = this->lost_stracks[i].tlwh[0] + this->lost_stracks[i].tlwh[2];
            float yb = this->lost_stracks[i].tlwh[1] + this->lost_stracks[i].tlwh[3];

            float yt_first = this->lost_stracks[i].begin_tlwh[1]; // 取初始位置最上方Y做判断条件

            yb = -yb;
            yt_first = -yt_first;
            // 两点式求直线, 注意这里对Y轴做了对称，由于数学坐标和YOLO图像坐标X同Y取反
            float yxl = (xl - point_begin.x) / (point_begin.x - point_end.x) * (-point_begin.y + point_end.y) -
                        point_begin.y;
            float yxr = (xr - point_begin.x) / (point_begin.x - point_end.x) * (-point_begin.y + point_end.y) -
                        point_begin.y;
            vector<float> tlbr = this->lost_stracks[i].tlwh;
            vector<float> be_tlbr = this->lost_stracks[i].begin_tlwh;
//            std::cout << "loss point lineY=" << lineY << " act=" << y << tlbr[0] << "\t" << tlbr[1] << "\t" << tlbr[2]
//                      << "\t" << tlbr[3] << "\tbegin_y="<< y_begin << std::endl;
            if (yb < std::min(yxl, yxr) && yt_first > std::min(yxr, yxl)) {
                std::cout << "you loss:" << this->lost_stracks[i].track_id << std::endl;
            }
        }
    }

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].state == TrackState::Tracked) {
            tracked_stracks_swap.push_back(this->tracked_stracks[i]);
        }
    }
    this->tracked_stracks.clear();
    this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
    this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

    //std::cout << activated_stracks.size() << std::endl;

    this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
    for (int i = 0; i < lost_stracks.size(); i++) {
        this->lost_stracks.push_back(lost_stracks[i]);
    }

    this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
    for (int i = 0; i < removed_stracks.size(); i++) {
        this->removed_stracks.push_back(removed_stracks[i]);
    }

    remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

    this->tracked_stracks.clear();
    this->tracked_stracks.assign(resa.begin(), resa.end());
    this->lost_stracks.clear();
    this->lost_stracks.assign(resb.begin(), resb.end());

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].is_activated) {
            output_stracks.push_back(this->tracked_stracks[i]);
        }
    }
    return output_stracks;
}