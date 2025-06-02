import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util


# GPU kontrolü
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU Available: {gpu_devices}")
else:
    print("GPU not available.")

# Kamera ayarları
cap = cv2.VideoCapture(0)

# Modelin indirilmesi için ayarlar
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# TensorFlow'un dondurulmuş model dosyasının yolu
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

OFFICE_CLASSES = [
    56,  # Chair
    62,  # Laptop
    66,  # Keyboard
    67,  # Cell Phone
    74,  # Book
    77,  # Scissors
    84,  # Clock
    72,  # Remote
    79,  # Oven (örneğin, mikrodalga olabilir)
    44,  # Bottle (su şişesi)
    46,  # Wine glass (kupa veya bardak)
    61,  # Dining table (masa)
    73,  # Vase (dekoratif obje)
    74,  # TV monitor
    78,  # Spoon (örneğin, çatal/bıçak olabilir)
    88   # Suitcase (çanta veya dosya kutusu)
]

# Modeli indirin
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
tar_file.extractall(path=os.getcwd())  # Güvenli yöntem
tar_file.close()

# Modeli yükleyin
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')

# Etiket haritasını yükleyin
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Nesne tespiti
with detection_graph.as_default():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, image_np = cap.read()
            if not ret:
                print("Kamera açılamadı.")
                break

            # Görseli genişletin
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Model ile tahmin yapın
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Sonuçları görselleştirin
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Sonucu gösterin
            cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))

            # Çıkış için 'q' tuşuna basın
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
