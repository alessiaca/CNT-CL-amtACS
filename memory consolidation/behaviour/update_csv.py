import numpy as np
import pandas

# Add the lines missing for the first 3 participants in the csv file for the day 2 recognition

block_order_file = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\behaviour\\data\\block_order.npy"
block_order = np.load(block_order_file)
file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\data\\4\\4_recognition2.csv"
data = pandas.read_csv(file_path)
header = list(data)

for n_par in [1,2,3]:

    # Read the data
    file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\data\\{n_par}\\{n_par}_recognition2.csv"
    block_order_tmp = block_order[n_par]
    data = pandas.read_csv(file_path)
    data = pandas.DataFrame.to_numpy(data)

    # Add the stimulation condition (column 2)
    conditions = np.array([block_order_tmp[int(i)] for i in data[:,0]])
    data = np.insert(data, 1, conditions, axis=1)

    # Add whether it was a bridge or house
    house_bridge = (data[:,3] >= 23) & (data[:,3] <= 32) | (data[:,3] >= 43) & (data[:,3] <= 52)
    house_bridge = np.array(house_bridge) + 1
    data = np.insert(data, 3, house_bridge, axis=1)

    # Add a line for correct/incorrect
    seen = data[:,4] <= 32 # 1 if presented during encoding
    correct = (seen == 1) & (data[:,5] == 1) | (seen == 0) & (data[:,5] == 0)
    data[:,6] = correct

    # Save them as a csv file
    df = pandas.DataFrame(data, columns=header)
    df.to_csv(f"data//{n_par}//{n_par}_recognition2_new.csv",index=False)

# Update the brdige/house line for the datasets 4-6
for n_par in [4,5,6]:
    # Read the data
    file_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\data\\{n_par}\\{n_par}_recognition2.csv"
    block_order_tmp = block_order[n_par]
    data = pandas.read_csv(file_path)
    data = pandas.DataFrame.to_numpy(data)

    # Update whether it was a bridge or house
    house_bridge = (data[:, 4] >= 23) & (data[:, 4] <= 32) | (data[:, 4] >= 43) & (data[:, 4] <= 52)
    house_bridge = np.array(house_bridge) + 1
    data[:,3] = house_bridge

    # Save them as a csv file
    df = pandas.DataFrame(data, columns=header)
    df.to_csv(f"data//{n_par}//{n_par}_recognition2_new.csv", index=False)