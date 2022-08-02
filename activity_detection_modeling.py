# ## Import Requirements
import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import eli5
import shap
from glob import glob
import time
import gc
import pickle
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import tensorflow_hub as hub


# ## Feature Extraction

# ### Feature Extraction Functions
# Get mediapipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Get tf object detection model
object_detection_model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1')

### Pose Estimation ###

def estimate_pose(image):
    # Setup mediapipe instance
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    try:
        return results.pose_landmarks.landmark
    except AttributeError:
        return None


def calculate_angle(first, mid, end):
    first = np.array(first) 
    mid = np.array(mid)  # Mid
    end = np.array(end)  # End

    radians = np.arctan2(end[1]-mid[1], end[0]-mid[0]) -         np.arctan2(first[1]-mid[1], first[0]-mid[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def calculate_all_angles(landmarks):
    angle_map = {
        'SHOULDER': ['ELBOW', 'SHOULDER', 'HIP'],
        'ELBOW': ['SHOULDER', 'ELBOW', 'WRIST'],
        'HIP': ['KNEE', 'HIP', 'SHOULDER'],
        'KNEE': ['HIP', 'KNEE', 'ANKLE'],
        'ANKLE': ['KNEE', 'ANKLE', 'PINKY']
    }
    
    angles = {}
    for angle_name, landmark_points in angle_map.items():
        for side in ['LEFT', 'RIGHT']:
            # get landmark names
            landmark_names = []
            for point in landmark_points:
                landmark_names.append(side+'_'+point)
            # calculate angle
            first = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].y]
            mid = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[1]).value].x,
                   landmarks[getattr(mp_pose.PoseLandmark, landmark_names[1]).value].y]
            end = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].x,
                   landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].y]
            visibility = np.mean([landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].visibility,
                          landmarks[getattr(mp_pose.PoseLandmark, landmark_names[1]).value].visibility,
                          landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].visibility])
            angle = calculate_angle(first, mid, end)
            angles[side+'_'+angle_name] = [angle]
            angles[side+'_'+angle_name+'_visibility'] = [visibility]

    angles = pd.DataFrame(angles)
    return angles


### Object Detection ###

def resize_image(image, dsize=(640, 640)):
    return cv2.resize(image, dsize=dsize, interpolation = cv2.INTER_CUBIC)

def detect_objects(image, model=object_detection_model):
  if image.shape != (640, 640):
    # Format for the Tensor
    image= resize_image(image)

  # To Tensor
  image_tensor  = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
  # Make detections
  detections = object_detection_model(image_tensor)
  detections = {key: value.numpy() for key, value in detections.items()}
  # Format results as dataframe
  df_result = pd.DataFrame({
    'class': detections['detection_classes'][0], 
    'detection_score': detections['detection_scores'][0], 
    'ymin': map(lambda x: x[0], detections['detection_boxes'][0]), 
    'xmin': map(lambda x: x[1], detections['detection_boxes'][0]), 
    'ymax': map(lambda x: x[2], detections['detection_boxes'][0]), 
    'xmax': map(lambda x: x[3], detections['detection_boxes'][0]),
  })
  # Filter necessary objects
  objects = {'laptop': df_result[df_result['class']==73],
              'keyboard': df_result[df_result['class']==76],
              'cellphone': df_result[df_result['class']==77],}

  return objects

### Feature Extraction ###

def sightline_intersects(ear, nose, obj_xmin, obj_ymin, obj_xmax, obj_ymax, img_shape):
  sightline = (nose[0]-ear[0],nose[1]-ear[1])
  current_point = (nose[0],nose[1])
  intersects = False
  while current_point[0] < img_shape[0] and current_point[1] < img_shape[1]       and current_point[0] > 0 and current_point[0] > 0 and not intersects:
      if current_point[0] < obj_xmax and current_point[1] < obj_ymax           and current_point[0] > obj_xmin and current_point[0] > obj_ymin:
          intersects = True
      else:
          current_point = (current_point[0] + sightline[0], current_point[1] + sightline[1])
  return intersects

def looks_at(image):
  looks_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # find objects user is looking at
    nose = landmarks[getattr(mp_pose.PoseLandmark, 'NOSE').value]
    sides = ['LEFT_', 'RIGHT_']
    for side in sides:
      ear = landmarks[getattr(mp_pose.PoseLandmark, side+'EAR').value]
      for obj, df in objects.items():
        for i in df.index:
          obj_row = df.loc[i]
          looks_at_[obj] = int((looks_at_[obj]==True) | sightline_intersects(
              [ear.x, ear.y], [nose.x, nose.y], obj_row.xmin, obj_row.ymin, obj_row.xmax, obj_row.ymax, image.shape))
  return pd.DataFrame({key: [value] for key, value in looks_at_.items()})


def hand_at(image):
  hand_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # find objects at hand
    fingers = ['LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX']
    for finger in fingers:
      finger = landmarks[getattr(mp_pose.PoseLandmark, finger).value]
      for obj, df in objects.items():
        for i in df.index:
          obj_row = df.loc[i]
          hand_at_[obj] = int((hand_at_[obj]==True) | (obj_row.xmin < finger.x and finger.x < obj_row.xmax and 
                                                       obj_row.ymin < finger.y and finger.y < obj_row.ymax))
  return pd.DataFrame({key: [value] for key, value in hand_at_.items()})

def focus_objects(image):
  looks_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  hand_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # iterate over objects
    for obj, df in objects.items():
      for i in df.index:
        obj_row = df.loc[i]
        
        # find objects at hand
        fingers = ['LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX']
        for finger in fingers:
          finger = landmarks[getattr(mp_pose.PoseLandmark, finger).value]
          hand_at_[obj] = int((hand_at_[obj]==True) | (obj_row.xmin < finger.x and finger.x < obj_row.xmax and 
                                                      obj_row.ymin < finger.y and finger.y < obj_row.ymax))
        
        # find objects user is looking at
        nose = landmarks[getattr(mp_pose.PoseLandmark, 'NOSE').value]
        sides = ['LEFT_', 'RIGHT_']
        for side in sides:
          ear = landmarks[getattr(mp_pose.PoseLandmark, side+'EAR').value]
          looks_at_[obj] = int((looks_at_[obj]==True) | sightline_intersects(
              [ear.x, ear.y], [nose.x, nose.y], obj_row.xmin, obj_row.ymin, obj_row.xmax, obj_row.ymax, image.shape))
          
  return pd.concat([pd.DataFrame({'hand_at_'+key: [value] for key, value in hand_at_.items()}),
                    pd.DataFrame({'looks_at_'+key: [value] for key, value in looks_at_.items()})], axis=1)

def extract_features(image):
  angles = None
  focus_objects_ = None
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # calculate all angles
    angles = calculate_all_angles(landmarks)
    # find what is at the hand and what is at the sightline
    focus_objects_ = focus_objects(image)
    # concat features
    features = pd.concat([angles, focus_objects_], axis=1)
    return features
  else:
    return None

def process_folder(folder_path, name, label, start_iter=0):
    # for each image in the folder
    img_paths = glob(folder_path+'/*.JPG') + glob(folder_path+'/*.jpg') + glob(folder_path+'/*.png') + glob(folder_path+'/*.PNG')
    n_process = 10
    n = len(img_paths)
    steps = int(n / n_process)
    for i in range(start_iter, steps):
        data = pd.DataFrame()
        iter_paths = img_paths[i*n_process:(i+1)*n_process]
        for j, img_path in enumerate(iter_paths):
            print(i*n_process+j, img_path)
            # read image
            image = cv2.imread(img_path)
            # extract features
            features = extract_features(image)
            del image
            gc.collect()
            data = pd.concat([data, features])
        data['label'] = label
        data.to_excel(os.path.join('feature-extraction', name+str(i)+'.xlsx'), index=False)
        del data
        gc.collect()
        print(f'Iter {i} completed!')


# ### Feature Extraction Job for Activity Detection

process_folder('activity-detection-data/object-detection-assets/non-working-no-angle', 'not-working', 0, start_iter=0)

process_folder('activity-detection-data/object-detection-assets/working-no-angle', 'working', 1, start_iter=0)

# concatanate all data
df = pd.DataFrame()
for p in glob('feature-extraction/*.xlsx'):
  df = pd.concat([df, pd.read_excel(p).reset_index(drop=True)])
df.to_excel('working-detection-pose+focusobjects.xlsx', index=False)


# ## Data Preperation
df_model = pd.read_excel('working-detection-pose+focusobjects.xlsx', engine='openpyxl')
X = df_model.drop(columns='label')
y = df_model['label']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# balance sides in the training set (mirror images)
X_mirrored = X_train.copy()
mirrored_cols = ['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 
       'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE'] 
mirrored_cols = [[c, c+'_visibility'] for c in mirrored_cols]
mirrored_cols = [item for sublist in mirrored_cols for item in sublist] 
mirrored_cols = mirrored_cols + ['looks_at_laptop', 'looks_at_keyboard', 'looks_at_cellphone', 'hand_at_laptop', 'hand_at_keyboard', 'hand_at_cellphone']
X_mirrored = X_mirrored[mirrored_cols]
X_mirrored.columns = X_train.columns
X_train = pd.concat([X_train, X_mirrored])
y_train = pd.concat([y_train, y_train])


# ## Try the first model
model = LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('First Results:', pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)), sep='\n')


# ## Model Selection
pipeline = Pipeline(steps=[("scaler", MinMaxScaler()), ("classifier", LGBMClassifier())])

params = [
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [LogisticRegression()],
      "classifier__C": [0.1, 1.0, 10.0, 100.0],
    },
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [RandomForestClassifier()],
      'classifier__max_depth': np.arange(1, 22, 2),
      'classifier__n_estimators': np.arange(10, 500, 50),
    },
    {
      'scaler': [StandardScaler(), MinMaxScaler()],
      'classifier': [LGBMClassifier()],
      'classifier__max_depth': np.arange(1, 52, 2),
      'classifier__num_leaves': np.arange(2, 203, 5),
      'classifier__n_estimators': np.arange(10, 501, 50),
      'classifier__learning_rate': np.arange(0.01, 1.502, 0.05)
    },
]

print('Tuning the model...')
search = RandomizedSearchCV(pipeline, params, n_iter=500, cv=10, random_state=42)
search.fit(X_train, y_train)

print('Best Estimator:', search.best_estimator_)
print('Best Score:', search.best_score_)


# ## Model Evaluation
model = Pipeline(steps=[('scaler', StandardScaler()),
                ('classifier',
                 LGBMClassifier(learning_rate=0.5, max_depth=11,
                                n_estimators=310, num_leaves=137))])

# uncomment to load the pretrained model
# model = pickle.load(open('activity_detection_model3.pkl', 'rb'))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Results of the Tuned Model:', pd.DataFrame(classification_report(y_test, y_pred>0.5, output_dict=True)), sep='\n')

# save the model
model_path = 'activity_detection_pose+focusobjects.pkl'
model_file = open(model_path, 'wb')
pickle.dump(model, model_file)
model_file.close()


# ## Model Explanation
feature_weights = eli5.explain_weights_df(model, feature_names=X_train.columns)
print('Results of the Tuned Model:', feature_weights, sep='\n')

# Fits the explainer
explainer = shap.Explainer(model.predict, X_test)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

pd.DataFrame(shap_values.values[y_test==0].mean(axis=0).reshape((1, 26)), columns = X_train.columns).T

pd.DataFrame(shap_values.values[y_test==1].mean(axis=0).reshape((1, 26)), columns = X_train.columns).T

