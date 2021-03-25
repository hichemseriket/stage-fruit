# import cv2
# import numpy as np
# import tensorflow as tf
#
# cap=cv2.VideoCapture(0)
import numpy as np
import os
import tensorflow as tf
import cv2

# ici je test un script de reconnaissance d'objet en essayant de ladapté a mon rseau et mon model , sauf que ce script utilmise selon moi une ia preentrainé et qui contient bcp de paramettre
#et beaucoup de fichier apparement car le fait de passé juste mon model etmes label suffit pas certe je me doutais quil fallait plus mais la je risque de perdre un temps fou a comprendre ce code a voir avec luc si faut poussé la dedans ou pas sachant que les script reconnaissance facial exite a foison
cap = cv2.VideoCapture(0)
labels = {
    # 1: "Fresh Banana",
    # 2: "Fresh Blueberry",
    # 3: "Fresh Huckleberry",
    # 4: "Fresh Orange",
    # 5: "Rotten Banana",
    # 6: "Rotten Blueberry",
    # 7: "Rotten Orange"
    1: "freshbanana",
    2: "freshblueberry",
    3: "freshhuckleberry",
    4: "freshorange",
    5: "rottenbanana",
    6: "rottenblueberry",
    7: "rottenorange"
}

MODEL_NAME = 'model.h5'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
color_infos = (255, 255, 0)

detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        # cap=cv2.VideoCapture('../../En_cours/tuto8-suite/Plan_9_from_Outer_Space_1959_512kb.mp4')
        cap = cv2.VideoCapture(0)
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            quit("Masque non géré")
        image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

        while True:
            ret, frame = cap.read()
            tickmark = cv2.getTickCount()
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(frame, 0)})
            nbr_object = int(output_dict['num_detections'])
            classes = output_dict['detection_classes'][0].astype(np.uint8)
            boxes = output_dict['detection_boxes'][0]
            scores = output_dict['detection_scores'][0]
            for objet in range(nbr_object):
                ymin, xmin, ymax, xmax = boxes[objet]
                if scores[objet] > 0.30:
                    height, width = frame.shape[:2]
                    xmin = int(xmin * width)
                    xmax = int(xmax * width)
                    ymin = int(ymin * height)
                    ymax = int(ymax * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_infos, 1)
                    txt = "{:s}:{:3.0%}".format(labels[classes[objet]], scores[objet])
                    cv2.putText(frame, txt, (xmin, ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1, color_infos, 2)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - tickmark)
            cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, color_infos, 2)
            cv2.imshow('image', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                for objet in range(500):
                    ret, frame = cap.read()
            if key == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
