import numpy as np
import pandas as pd

def preprocessing(coloc_dfs):
    '''
    small fn to automate some of the preprocessing steps that must be done
    on each df before we can concat: removing unncessary whitespaces,
    dropping rows without an intensity measurement
    '''
    # remove leading whitespace from col names
    for df in coloc_dfs:
        df.columns = [col.replace(' ', '') for col in df.columns]

    # drop rows without intensity measurement
    coloc_dfs = [df[~df['mean-background'].isna()] for df in coloc_dfs]

    # remove leading whitespace from filenames (note access via .loc to avoid setting with copy)
    for df in coloc_dfs:
        df.loc[:, 'filename'] = df.filename.str.replace(' ', '')

    return coloc_dfs
