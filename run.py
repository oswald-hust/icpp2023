from algs.scip import *
from tqdm import tqdm
import sys
import json

with open(sys.argv[1], 'r') as f:
    config = json.loads(f.read())

with open(config["output"], 'a') as output:
    for trace in config["traces"]:
        print("Test in trace: " + trace)
        output.write(trace+'\n')
        output.flush()
        for capacity in config["cache_sizes"]:
            print("Cache size: "+str(capacity)+"GB.")
            scip_test = SCIP(capacity * 1024 * 1024 * 1024) # GB->B
            dataSet = open(trace, "r")
            for line in tqdm(dataSet):
                lineSplit = line.strip("\n").split(" ") # time key size
                lineNum = lineSplit[0]
                key = lineSplit[1]
                size = int(lineSplit[2])
                scip_test.request(key, size)
            omr, bmr = scip_test.get_missrate()
            output.write('Cache size: '+str(capacity)+'GB, OMR(%): ' + str(omr) + ', BMR(%): ' + str(bmr) + '.\n')
            output.flush()
            dataSet.close()
output.close()
f.close()