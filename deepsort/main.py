import numpy as np
from IoU_Tracker import tracker
from Processing import postprocess
from scipy.spatial import distance

if __name__ == "__main__":

    camera = 74 # validation set
    # camera = 75 # test set

    # The number of people in the dataset is 5. Bonus points for method that does not require this line of hard coding 
    # number_of_people = 5
    number_of_people = None
    # confidence score threshold
    confidence_threshold = 0.38
    # tracklet length threshold
    trk_threshold = 100
    # output file path
    result_path = './deepsort/deepsort_result.txt'

    # Load the data
    detection = np.loadtxt('./detection.txt',delimiter=',',dtype=None)
    embedding = np.load('./embedding.npy', allow_pickle=True)
    inds = detection[:,0] == camera
    raw_detection = detection[inds]
    raw_embedding = embedding[inds]
    sort_inds = raw_detection[:, 1].argsort()
    raw_detection = raw_detection[sort_inds]
    raw_embedding = raw_embedding[sort_inds]

    # Filter out false detection based on confidence score, with a adjustable threshold
    # Detection format: <camera ID>, <Frame ID>, <class>, <x1>, <y1>, <x2>, <y2>, <confidence score>
    test_detection = []
    test_embedding = []
    count = 0
    for det, emb in zip(raw_detection, raw_embedding):
        score = det[7]
        if score > confidence_threshold:
            test_detection.append(det)
            test_embedding.append(emb)
        else:
            count += 1

    test_detection = np.array(test_detection)
    test_embedding = np.array(test_embedding)
    print(count, "number of deleted detection")

    mot = tracker()
    postprocessing = postprocess(number_of_people,'agglo')

    # Run the IoU tracking
    tracklets = mot.run(test_detection,test_embedding)
    for trk in tracklets:
        print('Track', trk.ID, 'length:', len(trk.times))

    tracklets = np.array([trk for trk in tracklets if len(trk.boxes) > trk_threshold])
    features = np.array([trk.final_features for trk in tracklets])
    
    # Run the Post Processing to merge the tracklets
    labels = postprocessing.run(features) # The label represents the final tracking ID, it starts from 0. We will make it start from 1 later.
    tracking_result = []
    tracking_out = []
    print(len(tracklets))
    print(labels)
    print('Writing Result ... ')
    for i,trk in enumerate(tracklets):
        trk_len = len(trk.boxes)
        final_tracking_id = labels[i]+1 # make it starts with 1
        pose_diff = np.linalg.norm(np.std(trk.pose, axis=0))
        print(pose_diff)
        print('Track', final_tracking_id, 'length:', trk_len, 'pose std:', pose_diff)
        if pose_diff > 50:
            for idx in range(len(trk.boxes)):
                frame = trk.times[idx]
                x1,y1,x2,y2 = trk.boxes[idx]
                x,y,w,h = x1,y1,x2-x1,y2-y1
                result = [camera,final_tracking_id,frame,x,y,w,h]
                tracking_result.append(result)
        else:
            print('Track', final_tracking_id, 'removed with a pose std of:', pose_diff)
    

    # sort result by frame id
    tracking_result = np.array(tracking_result)
    sort_inds = tracking_result[:, 2].argsort()
    tracking_result = tracking_result[sort_inds]

    for result in tracking_result:
        camera = result[0]
        final_tracking_id = result[1]
        frame = result[2]
        x, y, w, h = result[3:]
        result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera,final_tracking_id,frame,x,y,w,h)

        tracking_out.append(result)

    print('Save tracking results at {}'.format(result_path))

    with open(result_path,'w') as f:
        f.writelines(tracking_out)