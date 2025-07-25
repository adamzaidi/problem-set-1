'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def calibration_plot(y_true, y_prob, name, n_bins=5):
    true_prob, pred_prob = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(pred_prob, true_prob, marker="o", label=name)

def run_calibration_plot(df=None):
    if df is None:
        df = pd.read_csv("data/df_arrests_dt.csv")

    y = df["y"]

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.figure(figsize=(6, 6))
    calibration_plot(y, df["pred_lr"], "Logistic Regression")
    calibration_plot(y, df["pred_dt"], "Decision Tree")
    plt.legend()
    plt.title("Calibration Plot")

    plt.savefig("plots/calibration_plot.png")
    print("saved calibration plot to plots/")
