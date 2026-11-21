import os
from churn.config import ARTIFACTS_DIR
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from churn.transformation import Transformation
from churn.encoder import EncoderData
from churn.scale_data import ScaleData
from churn.utils import save_process
    
class TransformDataPipe:
    def __init__(self, data_path, transformer, scaler):
        self.process_path= os.path.join(ARTIFACTS_DIR, 'process', "transform_process.pkl")
        self.data_path: str= data_path
        self.transformer: str= transformer
        self.scaler = scaler
        
    def build_transform_data_pipeline(self):
            
            my_pipeline = Pipeline(steps = [ 
    
                  ('transformation', Transformation(self.transformer)),
                  ('encoder', EncoderData()),
                  ('scaledata', ScaleData(self.scaler))
                  ])
            
            # preprocessor=ColumnTransformer([
            #     ("num_pipeline",num_pipeline,numerical_columns),
            #     ("cat_pipelines",cat_pipeline,categorical_columns)
            #     ])

            return(my_pipeline)
            
    def transform_data_pipeline(self): 
            df=pd.read_csv(self.data_path)
            process_pipeline = self.build_transform_data_pipeline()
            df = process_pipeline.fit_transform(df)

            #df = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            df.to_csv(self.data_path,index=False,header=True)
           
            save_process(process_path=self.process_path, process=process_pipeline)   
            return(self.data_path)
