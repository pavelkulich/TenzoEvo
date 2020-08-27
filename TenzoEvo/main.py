import pandas as pd
import manipulator as dtm
import ga
import models

# pd.set_option('display.max_columns', None)

tenzometric = pd.read_csv('data/44t.asc', sep='\t', header=0, skiprows=[1])
t1 = tenzometric['T1']
t2 = tenzometric['T2']
t_avg = (t1 + t2) / 2 + 63

measured_data = pd.DataFrame(zip(tenzometric['x'], -t_avg), columns=['x_axis', 'y_axis'])

man = dtm.Manipulator(measured_data)
man.get_significant_points()
new_measured_data = man.get_measured_data()

model = models.Model('dynamic_double_pasternak')
gen_algs = ga.GA(model, new_measured_data, 5000, 50, man)
gen_algs.run_optimization()

print(gen_algs.get_init_population())
