import numpy as np
import pandas as pd

from R import *

pd = pd.read_csv(DATASET_CSV_FILE_PATH)
pd.to_csv("test_csv.gzip", index=False, compression='gzip')

