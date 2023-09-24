import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class Predictor:
  def __init__(self):
    self.model: tf.keras.Model | None = None
    self.scaler: MinMaxScaler | None = None
    
  def load(self):
    currentPath = os.getcwd()
    self.model = tf.keras.models.load_model(os.path.join(currentPath, "static/classify_tf.h5"))
    self.scaler = joblib.load(os.path.join(currentPath, "static/cluster_scaler.joblib"))
  
  def reduce_data(self, df):
    dfc = df.copy()
    dfc["P1"] = (dfc["Q1"]+dfc["Q2"]+dfc["Q3"]+dfc["Q4"]+dfc["Q5"]+dfc["Q6"]
                  +dfc["Q7"]+dfc["Q8"]+dfc["Q9"]+dfc["Q10"]+dfc["Q11"]+dfc["Q12"]
                  +dfc["Q13"]+dfc["Q14"]+dfc["Q15"]+dfc["Q16"]+dfc["Q17"]+dfc["Q18"]
                  +dfc["Q19"]+dfc["Q20"]+dfc["Q21"]+dfc["Q22"]+dfc["Q23"]+dfc["Q24"]
                  +dfc["Q25"]+dfc["Q26"]+dfc["Q27"]+dfc["Q28"]+dfc["Q29"]+dfc["Q30"]
                  +dfc["Q31"]+dfc["Q32"]+dfc["Q33"]+dfc["Q34"]+dfc["Q35"]+dfc["Q36"]
                  +dfc["Q37"]+dfc["Q38"]+dfc["Q39"]+dfc["Q40"]+dfc["Q41"]+dfc["Q42"]
                  +dfc["Q43"]+dfc["Q44"]+dfc["Q45"]+dfc["Q46"])

    dfc["P2"] = (dfc["Q47"]+dfc["Q48"]+dfc["Q49"]+dfc["Q50"]+dfc["Q51"]+dfc["Q51"]
                  +dfc["Q52"]+dfc["Q53"]+dfc["Q54"]+dfc["Q55"]+dfc["Q56"]+dfc["Q57"]
                  +dfc["Q58"]+dfc["Q59"]+dfc["Q60"]+dfc["Q61"]+dfc["Q62"]+dfc["Q63"]
                  +dfc["Q64"])

    dfc["P3"] = (dfc["Q65"]+dfc["Q66"]+dfc["Q67"]+dfc["Q68"]+dfc["Q69"]+dfc["Q70"])

    dfc["P4"] = (dfc["Q71"]+dfc["Q72"]+dfc["Q73"]+dfc["Q74"]+dfc["Q75"]+dfc["Q76"]
                  +dfc["Q77"]+dfc["Q78"]+dfc["Q79"])

    return dfc[["P1", "P2", "P3", "P4"]].values
  
  def predict(self, data):
    data = np.array(data).astype('int64').reshape(1, -1)
    data = pd.DataFrame(data, columns=['Q'+str(i) for i in range(1, 80)])

    data_reduced = self.reduce_data(data)
    data_scaled = self.scaler.transform(data_reduced)

    prediction = self.model.predict(data_scaled)
    prediction = np.argmax(prediction, axis=1)

    return prediction[0]