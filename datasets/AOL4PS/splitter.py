import numpy as np
import os
from pathlib import Path

def validation_split(validation_fraction=0.1):
    directory= str(Path(__file__).parent.absolute())
    if(os.path.exists(directory+"/validation_data.csv")):
        print("Existing validation split found.")
        return
    np.random.seed(143)
    num_lines=0
    with open(directory+"/data.csv", "rb") as f: #faster to do outside since we can use rb mode
        num_lines = sum(1 for _ in f)
    print("Found",num_lines,"training log samples")
    val_array=np.full(num_lines-1,False,dtype=bool)
    val_array[:int(np.ceil((num_lines-1)*validation_fraction))]=True
    np.random.shuffle(val_array)
    
    with open(directory+"/data.csv","r") as f, open(directory+"/validation_data.csv","w") as val_f, open(directory+"/training_data.csv","w")as tr_f:
        val_f.truncate(0) #cleaning values
        tr_f.truncate(0) #cleaning values
        for i,line in enumerate(f):
            if(i==0):
                tr_f.write(line)
                val_f.write(line)
                continue
            if val_array[i-1]:
                val_f.write(line)
            else:
                tr_f.write(line)