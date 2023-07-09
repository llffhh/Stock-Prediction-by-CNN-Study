from Technical_Index import TechIndicator
import pandas as pd
import numpy as np


df = pd.read_csv('../2330.TW.csv')

print(df)

TechIndicator = TechIndicator()
print(TechIndicator.RSI(df,7))