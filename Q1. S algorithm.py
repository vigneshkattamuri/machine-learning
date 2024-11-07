import pandas as pd
def find_s_algorithm(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    features = data.columns[:-1]
    target = data.columns[-1]
    hypothesis = None
    for index, row in data.iterrows():
        if row[target] == "Yes": 
            if hypothesis is None:
                hypothesis = row[features].values
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row[features].values[i]:
                        hypothesis[i] = '?'
    print("The most specific hypothesis found by FIND-S:")
    print(hypothesis)
file_path = 'C:/Users/kadiy/Documents/ML Practical/training_data.csv'
find_s_algorithm(file_path)
