# This is the baseline code for the single camera tracker using bounding box IoU (intersection over union)
import numpy as np
from numpy.linalg import inv
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from KalmanFilter import KalmanFilter

# calculate the overlap ratio of two bounding boxes
def calculate_iou(bbox1, bbox2):

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    return iou

# Convert bounding box to format `(center x, center y, aspect ratio, height)`, 
#  where the aspect ratio is `width / height`.
def to_xyah(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    a = width/height
    center_x = (x2 + x1)/2
    center_y = (y2 + y1)/2 
    ret = [center_x, center_y, a, height]
    return np.array(ret)

def get_center(box):
    x1, y1, x2, y2 = box
    center_x = (x2 + x1)/2
    center_y = (y2 + y1)/2 
    return [center_x, center_y]

# base class for tracklet
class tracklet:
    def __init__(self,tracking_ID,box,feature,time, mean, covariance):
        self.ID = tracking_ID
        self.boxes = [box]
        self.features = [feature]
        self.times = [time]
        self.cur_box = box
        self.cur_feature = feature
        self.K = 150
        self.last_k_feature = []
        self.alive = True
        self.final_features = feature
        self.mean = mean
        self.covariance = covariance
        self.miss_count = 0
        self.pose = [get_center(box)]
        self.confirmed = False
    
    def update(self, box, feature, time, kf):
        self.cur_box = box
        self.boxes.append(box)
        if len(self.last_k_feature) > self.K:
            self.last_k_feature.pop()
        self.last_k_feature.append(feature)
        self.cur_feature = sum(self.last_k_feature)/len(self.last_k_feature)
        self.features.append(feature)
        self.times.append(time)
        measurement = to_xyah(box)
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)
        self.is_confirmed()
        self.miss_count = 0
        self.pose.append(get_center(box))


    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def close(self):
        self.alive = False
    
    def is_confirmed(self):
        if len(self.times) > 3 and not self.confirmed:
            self.confirmed = True

    def missed(self):
        self.miss_count += 1

    def get_avg_features(self):
        self.final_features = sum(self.features)/len(self.features) # we do the average pooling for the final features


# class for multi-object tracker
class tracker:
    def __init__(self):
        self.all_tracklets = []
        self.cur_tracklets = []
        self.kf = KalmanFilter()
        self.max_miss = 900

    def run(self, detections, features):
        # 18010
        
        for frame_id in range(0,18010):
            if frame_id%1000 == 0:
                print('Tracking | cur_frame {} | total frame 18010'.format(frame_id))

            inds = detections[:,1] == frame_id
            cur_frame_detection = detections[inds]
            cur_frame_features = features[inds]

            for track in self.cur_tracklets:
                track.predict(self.kf)

            # no tracklets in the first frame
            if len(self.cur_tracklets) == 0:
                for idx in range(len(cur_frame_detection)):
                    self.initialize_track(cur_frame_detection[idx][3:7], cur_frame_features[idx], frame_id)
            else:
                confirmed_tracklets = [trk for trk in self.cur_tracklets if trk.confirmed]

                cost_matrix = np.zeros((len(confirmed_tracklets),len(cur_frame_detection)))

                for i in range(len(confirmed_tracklets)):
                    for j in range(len(cur_frame_detection)):
                        # Compute current tracklet and detection center
                        measurement = to_xyah(cur_frame_detection[j][3:7])
                        mean = confirmed_tracklets[i].mean
                        covariance = confirmed_tracklets[i].covariance
                        mal_dist = self.kf.gating_distance(mean, covariance, measurement)
                        emb_score = distance.cosine(confirmed_tracklets[i].cur_feature, cur_frame_features[j])
                        cost_matrix[i][j] = emb_score*0.999 + mal_dist*0.001
                
                if len(confirmed_tracklets) != 0:
                    row_inds,col_inds = linear_sum_assignment(cost_matrix)
                else:
                    row_inds,col_inds = [], []
    
                matches = min(len(row_inds),len(col_inds))
                for idx in range(matches):
                    row,col = row_inds[idx],col_inds[idx]
                    confirmed_tracklets[row].update(cur_frame_detection[col][3:7], cur_frame_features[col], frame_id, self.kf)

                
                unmatched_detection = [det for idx,det in enumerate(cur_frame_detection) if idx not in col_inds]
                unmatched_features = [feat for idx,feat in enumerate(cur_frame_features) if idx not in col_inds]
                unmatched_tracklets = [trk for idx,trk in enumerate(confirmed_tracklets) if idx not in row_inds]
                unconfirmed_tracklets = [trk for trk in self.cur_tracklets if not trk.confirmed]
                tracklets = unmatched_tracklets + unconfirmed_tracklets

         
                cost_matrix = np.zeros((len(tracklets),len(unmatched_detection)))
                for i in range(len(tracklets)):
                    for j in range(len(unmatched_detection)):
                        # Compute current tracklet and detection center
                        iou_score = 1 - calculate_iou(tracklets[i].cur_box, unmatched_detection[j][3:7])
                        cost_matrix[i][j] = iou_score

                
                if len(tracklets) != 0:
                    row_inds,col_inds = linear_sum_assignment(cost_matrix)
                else:
                    row_inds,col_inds = [], []

                matches = min(len(row_inds),len(col_inds))
                    
                for idx in range(matches):
                    row,col = row_inds[idx],col_inds[idx]
                    if cost_matrix[row,col] > 0.6 and matches == 1:
                        if tracklets[row].confirmed and tracklets[row].miss_count < self.max_miss:
                            tracklets[row].missed()
                        else:
                            print('Deleted track:', tracklets[row].ID)
                            tracklets[row].close()
                        self.initialize_track(unmatched_detection[col][3:7], unmatched_features[col], frame_id)
                    else:
                        tracklets[row].update(unmatched_detection[col][3:7], unmatched_features[col], frame_id, self.kf)
                
                # initiate unmatched detections as new tracklets
                for idx,det in enumerate(unmatched_detection):
                    if idx not in col_inds: # if it is not matched in the above Hungarian algorithm stage
                        self.initialize_track(det[3:7], unmatched_features[idx], frame_id)
                
                
                # loop through unmatched tracks and check condition for deletion
                for idx, trk in enumerate(tracklets):
                    if idx not in row_inds:
                        if trk.confirmed and trk.miss_count < self.max_miss:
                            trk.missed()
                        else:
                            print('Deleted track:', trk.ID)
                            trk.close()
                

            self.cur_tracklets = [trk for trk in self.cur_tracklets if trk.alive]            

        final_tracklets = self.all_tracklets

        # calculate an average final features (512x1) for all the tracklets
        for trk_id in range(len(final_tracklets)):
            final_tracklets[trk_id].get_avg_features()

        return final_tracklets
    
    def initialize_track(self, box, feature, frame_id):
        mean, covariance = self.kf.initiate(to_xyah(box))
        new_tracklet = tracklet(len(self.all_tracklets)+1, box, feature, frame_id,
                                mean, covariance)
        self.cur_tracklets.append(new_tracklet)
        self.all_tracklets.append(new_tracklet)