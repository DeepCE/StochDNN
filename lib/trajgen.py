import random
import pandas as pd



param_dict = {
    "ll": [0.0491, 0.0220, 0.1447, 0.0525, 0.0917, 0.2500, 0.9076, 0.9155],
    "mm": [0.0495, 0.0215, 0.1489, 0.0563, 0.0456, 0.3607, 0.8920, 0.8990]
}



def model_to_generate_trajectory(_type, n_traj, n_obs, first_log_price):
    b = param_dict[_type]
    matrix = []
    for i in range(n_traj):
        x = [first_log_price]
        regime = "stable"

        for j in range(1, n_obs+1):
            if regime == "stable":
                if random.uniform(0, 1) < b[6]:
                    point = (1 - b[0]) * x[j-1] + b[1] * random.gauss(0, 1)
                else:
                    if random.uniform(0, 1) < 1-b[4]:
                        point = (1-b[2]) * x[j-1] + b[3] * random.gauss(0, 1)
                    else:
                        point = (1-b[2]) * x[j-1] + b[3] * random.gauss(0, 1) + b[5] * random.gauss(0, 1)
                    regime = "excited"
            else:
                if random.uniform(0, 1) < b[7]:
                    if random.uniform(0, 1) < 1-b[4]:
                        point = (1-b[2]) * x[j-1] + b[3] * random.gauss(0, 1)
                    else:
                        point = (1-b[2]) * x[j-1] + b[3] * random.gauss(0, 1) + b[5] * random.gauss(0, 1)
                else:
                    point = (1-b[0]) * x[j-1] + b[1] * random.gauss(0, 1)
                    regime = "stable"
            x.append(point)
        matrix.append(x[1:])
    return matrix