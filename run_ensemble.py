import glob
import pandas as pd
import datetime
import os
import sys

def load_prob_csv(path):
    prob_path = glob.glob(path + "/*prob*.csv")[0]
    df = pd.read_csv(prob_path, header=None, names=("id", "prob"), index_col=0)
    return df

def ensumble(file_paths):
    names = []
    result = None
    for file_path in file_paths:
        names.append(file_path.split("/")[-1])
        df = load_prob_csv(file_path)
        if result is None:
            result = pd.DataFrame(index=df.index)
            result["prob"] = df["prob"]
        else:
            result["prob"] += df["prob"]
    result["prob"] /= len(file_paths)  
    result["result"] = result["prob"].map(round)
    del result["prob"]
    return result, "-".join(names)

if __name__ == "__main__":
    dt_now = datetime.datetime.now()
    now_str = dt_now.strftime('%Y%m%d_%H%M%S')
    output_path = "./outputs/ensemble/{}".format(now_str)
    os.makedirs(output_path)
    df, name = ensumble(sys.argv[1:])
    df.to_csv("{}/ensemble_result_{}.csv".format(output_path, name), header=False)