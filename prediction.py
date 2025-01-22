import pandas as pd
import imblearn
import pickle
from pycaret.classification import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from utils import *
import argparse



def generate_predictions(args):
    
    # read data
    df = pd.read_csv(args.infile)
    
    # Load custom pipeline
    with open(args.cust_pipe, "rb") as f:
        cust_pipeline = pickle.load(f)
    
    # Load pycaret pipeline
    load_config(args.pycaret_pipe)
    
    # Load model
    saved_model = load_model(args.model.split('.')[0])
    
    # Custom transformations
    df_new = cust_pipeline.transform(df)
    
    # Generate predictions
    df_pred = predict_model(saved_model, data=df_new)
    
    # Save predictions
    df_pred.to_csv(args.outfile)
    

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="CSV file containing data for prediction")
    parser.add_argument("cust_pipe", help="PKL file containing the custom pipeline")
    parser.add_argument("pycaret_pipe", help="PKL file containing pycaret pipeline")
    parser.add_argument("model", help="PKL file containing model for predictions")
    parser.add_argument("outfile", help="CSV file with model predictions")
    args = parser.parse_args()

    generate_predictions(args)

    
    
if __name__ == '__main__':
    main()