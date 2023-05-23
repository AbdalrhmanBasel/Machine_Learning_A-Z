import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('data/ex2data1.txt', sep=',', header=None)
    df.columns = ['exam_score_1', 'exam_score_2', 'label']

    df.describe().T