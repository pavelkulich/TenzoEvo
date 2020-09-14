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
t_avg = ((t1 + t2) / 2 + 63) * 5

measured_data = pd.DataFrame(zip(tenzometric['x'], -t_avg), columns=['x_axis', 'y_axis'])

man = dtm.Manipulator(measured_data)
man.get_significant_points()
new_measured_data = man.get_measured_data()

model = models.Model('dynamic_double_pasternak')
gen_algs = ga.GA(model, new_measured_data, 320, 10, man)
gen_algs.run_optimization()
