import models
import ga_config as cfg
import pandas as pd
import matplotlib.pyplot as plt

model = models.Model('dynamic_double_pasternak', True)

params_list = cfg.DYNAMIC_DOUBLE_PASTERNAK_TEST

head = []
value = []
for param in params_list:
    head.append(param[0])
    value.append(param[1])

params = pd.Series(value, index =head)

analytical_data = model.calculate_model(params, 225000)

plt.plot(analytical_data['x_axis'], analytical_data['y_axis'])
plt.title(f"min value: {min(analytical_data['y_axis'])}")
plt.grid()
plt.show()