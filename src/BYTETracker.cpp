#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer, int max_frame) {
    track_thresh = 0.5;
    high_thresh = 0.6;
    match_thresh = 0.8;
    this->frame_count = max_frame;
    frame_id = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

BYTETracker::~BYTETracker() {
}

vector<STrack> BYTETracker::update(const vector<Detection> &objects) {

    ////////////////// Step 1: Get detections //////////////////
    this->frame_id++;
    vector<STrack> update_activated;
    vector<STrack> upload_refine;
    vector<STrack> update_remove;
    vector<STrack> update_loss;
    vector<STrack> detections;
    vector<STrack> detections_low;

    vector<STrack> detections_cp;
    vector<STrack> tracked_stracks_swap;
    vector<STrack> resa, resb;
    vector<STrack> output_stracks;

    vector<STrack *> unconfirmed;
    vector<STrack *> tracked_stacks;
    vector<STrack *> strack_pool;
    vector<STrack *> r_tracked_stracks;

    // 将最新的标注信息输入
    if (!objects.empty()) {
        for (int i = 0; i < objects.size(); i++) {
            vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = objects[i].rect.x;
            tlbr_[1] = objects[i].rect.y;
            tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
            tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

            float score = objects[i].prob;

            STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, int(objects[i].label));
            if (score >= track_thresh) {
                detections.push_back(strack);
            } else {
                detections_low.push_back(strack);
            }

        }
    }

    // detections中包含高分标注，detections_low保存最新的低分标注
    // Add newly detected tracklets to tracked_stacks
    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (!this->tracked_stracks[i].is_activated) {
            // 首次调用update时，全部将加入到此数组
            unconfirmed.push_back(&this->tracked_stracks[i]);
        } else {
            tracked_stacks.push_back(&this->tracked_stracks[i]);
        }
    }


    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stacks, this->lost_stracks);
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
            update_activated.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            upload_refine.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    detections_cp.reserve(u_detection.size());
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
            update_activated.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            upload_refine.push_back(*track);
        }
    }

    for (int i = 0; i < u_track.size(); i++) {
        STrack *track = r_tracked_stracks[u_track[i]];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            update_loss.push_back(*track);
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
        update_activated.push_back(*unconfirmed[matches[i][0]]);
    }

    for (int i = 0; i < u_unconfirmed.size(); i++) {
        STrack *track = unconfirmed[u_unconfirmed[i]];
        track->mark_removed();
        update_remove.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        STrack *track = &detections[u_detection[i]];
        if (track->score < this->high_thresh) {
            continue;
        }
        track->activate(this->kalman_filter, this->frame_id);
        update_activated.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (int i = 0; i < this->lost_stracks.size(); i++) {
        if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost) {
            this->lost_stracks[i].mark_removed();
            update_remove.push_back(this->lost_stracks[i]);
        }
    }

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].state == TrackState::Tracked) {
            tracked_stracks_swap.push_back(this->tracked_stracks[i]);
        }
    }
    this->tracked_stracks.clear();
    this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks = joint_stracks(this->tracked_stracks, update_activated);
    this->tracked_stracks = joint_stracks(this->tracked_stracks, upload_refine);

    this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
    for (int i = 0; i < update_loss.size(); i++) {
        this->lost_stracks.push_back(update_loss[i]);
    }

    this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
    for (int i = 0; i < update_remove.size(); i++) {
        this->removed_stracks.push_back(update_remove[i]);
    }

    remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

    this->tracked_stracks.clear();
    this->tracked_stracks.assign(resa.begin(), resa.end());
    this->lost_stracks.clear();
    this->lost_stracks.assign(resb.begin(), resb.end());

    if (this->frame_id >= this->frame_count) {
        pop_target();
    }

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].is_activated) {
            output_stracks.push_back(this->tracked_stracks[i]);
            this->record_all_tracked.push_back(this->tracked_stracks[i]);
        }
    }
    return output_stracks;
}

void BYTETracker::pop_target() {
    map<int, STrack *> min_bbox;
    map<int, STrack *> max_bbox;
    for (int i = 0; i < this->record_all_tracked.size(); ++i) {
        const int l = this->record_all_tracked[i].track_id;
        if (!min_bbox.count(l)) {
            min_bbox[l] = &this->record_all_tracked[i];
        } else {
            bool b = this->record_all_tracked[i].frame_id < min_bbox.find(l)->second->frame_id;
            if (b) {
                STrack sTrack = this->record_all_tracked[i];
                min_bbox[l] = &sTrack;
            }
        }
        if (!max_bbox.count(l)) {
            max_bbox[l] = &this->record_all_tracked[i];
        } else {
            bool b = this->record_all_tracked[i].frame_id > max_bbox.find(l)->second->frame_id;
            if (b) {
                max_bbox[l] = &this->record_all_tracked[i];
            }
        }
    }

    for (auto p: max_bbox) {
        // 用下角的点，判断商品拿取,右开门肯定是左下，左开门肯定是右下，取两者最小值能最大限度地判断商品出界
        STrack *st_end = p.second;
        STrack *st_begin = min_bbox.find(p.first)->second;
        float xr = st_end->tlwh[0] + st_end->tlwh[2];
        float xl = st_end->tlwh[0] + st_end->tlwh[2];
        float yb = st_end->tlwh[1] + st_end->tlwh[3];

        float yt_first = st_begin->tlwh[1]; // 取初始位置最上方Y做判断条件

        yb = -yb;
        yt_first = -yt_first;
        // 两点式求直线, 注意这里对Y轴做了对称，由于数学坐标和YOLO图像坐标X同Y取反
        float yxl = (xl - point_begin.x) / (point_begin.x - point_end.x) * (-point_begin.y + point_end.y) -
                    point_begin.y;
        float yxr = (xr - point_begin.x) / (point_begin.x - point_end.x) * (-point_begin.y + point_end.y) -
                    point_begin.y;
        vector<float> tlbr = st_end->tlwh;
        vector<float> be_tlbr = st_begin->tlwh;
//        std::cout << "loss point lineY=" << std::min(yxl, yxr) << " act=" << yb << tlbr[0] << "\t" << tlbr[1] << "\t"
//                  << tlbr[2]
//                  << "\t" << tlbr[3] << "\tbegin_y=" << yt_first << "\tminiY=" << std::min(yxr, yxl) << std::endl;
        if (yb < std::min(yxl, yxr) && yt_first > std::min(yxr, yxl)) {
            std::cout << "you loss:" << st_end->track_id << std::endl;
        } else if (yb > std::min(yxl, yxr) && yt_first < std::min(yxr, yxl)) {
            std::cout << "you put:" << st_end->track_id << endl;
        } else {
            std::cout << "dont known" << st_end->track_id << std::endl;
        }
    }
}



//    if (this->frame_id >= this->frame_count) {
//        for (int i = 0; i < this->tracked_stracks.size(); ++i) {
//            std:cout << "ok="<< this->tracked_stracks[i].track_id<<endl;
//        }
//    }
