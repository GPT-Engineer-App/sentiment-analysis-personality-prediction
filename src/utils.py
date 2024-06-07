import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    train_df, val_df = train_test_split(df, test_size=0.2)
    return train_df, val_df