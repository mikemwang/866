import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
import sys

fig, ax = plt.subplots()
with open('experimental_data/data.yaml', 'r') as f:
    data_file = yaml.load(f)
    for experiment in data_file:
        #if experiment == sys.argv[1]:
        if experiment in ["test_movies/ttc_no_features.mov",
                          "test_movies/ttc_some_features.mov",
                          "test_movies/ttc_many_features.mov"]:
            color = [1,0,0,0.6] if experiment == "test_movies/ttc_no_features.mov" else [0.5,1,0.5,0.6] if experiment == "test_movies/ttc_some_features.mov" else [0,1,0,0.6]
            for i in ["NO_FOE"]:
                if data_file[experiment][i]:
                    ax.plot(data_file[experiment][i]['frames'][5:],data_file[experiment][i]['data'][5:],
                            color=color)
                    ax.plot(data_file[experiment][i]['frames'][5:],data_file[experiment][i]['groundtruth'][5:], color=[0,0,1])

                    a = np.array([data_file[experiment][i]['data'][5:]])
                    b = np.array([data_file[experiment][i]['groundtruth'][5:]])
                    print(experiment + " " + i + " average error: " + str(np.sum(a-b)/float(a.size)))

plt.show()
