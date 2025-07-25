'''
You will run this problem set from main.py, so set things up accordingly
'''
#as a note, i edited the names on my machine to part1_etl.py, part2_preprocesing.py, etc.
import pandas
from part1_etl import run_etl
from part2_preprocessing import run_preprocessing
from part3_logistic_regression import run_logistic_regression
from part4_decision_tree import run_decision_tree
from part5_calibration_plot import run_calibration_plot

def main():
    run_etl()
    df = run_preprocessing()
    df = run_logistic_regression(df)
    df = run_decision_tree(df)
    run_calibration_plot(df)

if __name__ == "__main__":
    main()
