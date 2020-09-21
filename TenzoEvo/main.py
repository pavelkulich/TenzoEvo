import pandas as pd
import manipulator as dtm
import ga
import models
import os
import shutil


def remove_folder_content(dir_name):
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        try:
            shutil.rmtree(file_path)
        except OSError:
            os.remove(file_path)


remove_folder_content('log')
remove_folder_content('plots')

# pd.set_option('display.max_columns', None)

tenzometric = pd.read_csv('data/44t.asc', sep='\t', header=0, skiprows=[1])
t1 = tenzometric['T1']
t2 = tenzometric['T2']

epsilon = ((t1 + t2) + 125) / 1000000
E = 210e9
W = 330e-6
M = E * W * epsilon

measured_data = pd.DataFrame(zip(tenzometric['x'], -M), columns=['x_axis', 'y_axis'])

man = dtm.Manipulator(measured_data)
man.get_significant_points()
new_measured_data = man.get_measured_data()

model = models.Model('dynamic_double_pasternak')
gen_algs = ga.GA(model, new_measured_data, 640, 5, man)
gen_algs.run_optimization()
