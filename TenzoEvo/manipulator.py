import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate
import matplotlib.pyplot as plt


class Manipulator:
    def __init__(self, measured_data):
        self.measured_data = measured_data
        self.measured_data.columns = ['x_axis', 'y_axis']
        self.measured_sampling_interval = np.abs(self.measured_data['x_axis'][1] - self.measured_data['x_axis'][0])
        self.analytical_data = None
        self.analytical_sampling_interval = None
        self.superposed_analytical_data = None
        self.superposed_analytical_data_resampled = None
        self.q_vector = pd.DataFrame()

    def set_analytical_data(self, analytical_data):
        self.analytical_data = analytical_data
        self.analytical_sampling_interval = np.abs(
            self.analytical_data['x_axis'][1] - self.analytical_data['x_axis'][0])
        # print(self.analytical_data)

    def get_significant_points(self, order=10, tolerance=130):
        self.measured_data['min'] = \
            self.measured_data.iloc[argrelextrema(self.measured_data['y_axis'].values, np.less_equal, order=order)[0]][
                'y_axis']
        self.measured_data['min'] = self.measured_data['min'].fillna(0)
        self.measured_data['min'] = self.measured_data.iloc[np.abs(self.measured_data['min'].values) >= tolerance][
            'min']

        # for presentation purpose
        # plt.plot(self.measured_data['x_axis'], self.measured_data['y_axis'])
        # plt.scatter(self.measured_data['x_axis'], self.measured_data['min'], color='r')
        # plt.grid(True)
        # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95)
        # plt.xlabel('time [s]', fontsize=15)
        # plt.ylabel('deflection [mm]', fontsize=15)
        # plt.show()

    def get_measured_data(self):
        return self.measured_data

    def adjust_q_vector(self):
        for i in range(self.measured_data['y_axis'][self.measured_data['min'] < 0].shape[0]):
            local_min = self.measured_data['y_axis'][self.measured_data['min'] < 0].iloc[i]
            q = local_min / np.min(self.analytical_data['y_axis'])
            self.q_vector[f'Q_{10 + i}'] = pd.DataFrame(columns=[f'Q_{i}'], data=[q])[f'Q_{i}']

    def get_adjusted_q_vector(self):
        return self.q_vector

    def get_superposed(self):
        if self.superposed_analytical_data is None:
            print(f'Superposed data equals {self.superposed_analytical_data}')
        return self.superposed_analytical_data

    def get_superposed_resampled(self):
        if self.superposed_analytical_data_resampled is None:
            print(f'Superposed resampled data equals {self.superposed_analytical_data_resampled}')
        return self.superposed_analytical_data_resampled

    def move_and_superpose(self):
        if self.analytical_data is not None:
            if 'min' in self.measured_data.columns:
                measured_mins = self.measured_data[self.measured_data['min'] < 0]
                counter = 1
                data_list = []
                for _, row in measured_mins.iterrows():
                    moved_analytical_data = self.__get_new_x_axis(row)
                    moved_analytical_data['y_axis'] = moved_analytical_data['y_axis'] * \
                                                      self.q_vector[f'Q_{10 + counter - 1}'][0]

                    if counter != 1:
                        idx = moved_analytical_data.iloc[(prev_moved_analytical_data['x_axis'] - np.min(
                            moved_analytical_data['x_axis'])).abs().argsort()[:1]].index[0]
                        moved_analytical_data = moved_analytical_data.set_index(
                            moved_analytical_data.index + idx + prev_moved_analytical_data.idxmin()[0])

                    prev_moved_analytical_data = moved_analytical_data.copy()
                    moved_analytical_data.columns = [f'x_axis_{counter}', f'y_axis_{counter}']
                    data_list.append(moved_analytical_data)
                    # plt.plot(moved_analytical_data[f'x_axis_{counter}'], moved_analytical_data[f'y_axis_{counter}'])
                    counter += 1

                # plt.grid(True)
                # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.95)
                # plt.xlabel('time [s]', fontsize=15)
                # plt.ylabel('deflection [mm]', fontsize=15)
                # plt.plot(self.measured_data['x_axis'], self.measured_data['y_axis'], color='#1f7bb8')
                # plt.show()

                # multi array by x axis
                presup_analytical_data = pd.concat(data_list, axis=1)

                # presup_analytical_data.to_csv('presup.csv')

                # create superposed dataframe
                sup_analytical_data = pd.DataFrame()

                for iter in range(1, counter):
                    if iter != 1:
                        sup_analytical_data['x_axis'] = sup_analytical_data['x_axis'].combine_first(
                            presup_analytical_data[f'x_axis_{iter}'])
                        sup_analytical_data['y_axis'] = sup_analytical_data['y_axis'] + presup_analytical_data[
                            f'y_axis_{iter}'].fillna(0)
                    else:
                        sup_analytical_data['x_axis'] = presup_analytical_data[f'x_axis_{iter}']
                        sup_analytical_data['y_axis'] = presup_analytical_data[f'y_axis_{iter}'].fillna(0)

                # crop tails
                self.superposed_analytical_data, self.measured_data = self.__crop_tails(sup_analytical_data)

                # resample data
                self.superposed_analytical_data_resampled = self.__resample_data()

            else:
                self.get_significant_points()
                self.move_and_superpose()
        else:
            print('Please set analytical data')

    def __crop_tails(self, sup_analytical_data):
        min_analytical = np.min(sup_analytical_data['x_axis'])
        max_analytical = np.max(sup_analytical_data['x_axis'])
        min_measured = np.min(self.measured_data['x_axis'])
        max_measured = np.max(self.measured_data['x_axis'])

        # analytická kratší na obou stranách
        if min_analytical >= min_measured and max_analytical <= max_measured:
            measured_data = self.measured_data.drop(self.measured_data[
                                                        self.measured_data['x_axis'] < min_analytical].index).drop(
                self.measured_data[
                    self.measured_data['x_axis'] > max_analytical].index).reset_index(drop=True).copy()
            # print('cut 1')

        # analytická delší na obou stranách
        elif min_analytical < min_measured and max_analytical > max_measured:
            sup_analytical_data = sup_analytical_data.drop(sup_analytical_data[
                                                               sup_analytical_data['x_axis'] < np.min(
                                                                   self.measured_data['x_axis'])].index).drop(
                sup_analytical_data[
                    sup_analytical_data['x_axis'] > np.max(self.measured_data['x_axis'])].index).reset_index(
                drop=True).copy()

            measured_data = self.measured_data.reset_index(drop=True).copy()
            # print(min_measured)
            # print(max_measured)
            # print(min_analytical)
            # print(max_analytical)
            # print(sup_analytical_data)
            # print('cut 2')

        # analytická kratší na začátku, delší na konci
        elif min_analytical >= min_measured and max_analytical > max_measured:
            measured_data = self.measured_data.drop(self.measured_data[
                                                        self.measured_data['x_axis'] < np.min(
                                                            sup_analytical_data['x_axis'])].index).reset_index(
                drop=True).copy()
            sup_analytical_data = sup_analytical_data.drop(sup_analytical_data[
                                                               sup_analytical_data['x_axis'] > np.max(
                                                                   self.measured_data['x_axis'])].index).reset_index(
                drop=True).copy()
            # print('cut 3')

        # analytická delší na začátku, kratší na konci
        elif min_analytical <= min_measured and max_analytical < max_measured:
            sup_analytical_data = sup_analytical_data.drop(sup_analytical_data[
                                                               sup_analytical_data['x_axis'] < np.min(
                                                                   self.measured_data['x_axis'])].index).reset_index(
                drop=True).copy()
            measured_data = self.measured_data.drop(self.measured_data[
                                                        self.measured_data['x_axis'] > np.max(
                                                            sup_analytical_data['x_axis'])].index).reset_index(
                drop=True).copy()
            # print('cut 4')

        else:
            print('this is the script error')
            print(min_measured)
            print(max_measured)
            print(min_analytical)
            print(max_analytical)

        return sup_analytical_data, measured_data

    def __get_new_x_axis(self, row):
        analytical_min = self.analytical_data[
            self.analytical_data['y_axis'] == self.analytical_data['y_axis'].min()]
        first_index = 0
        last_index = self.analytical_data.index[-1]
        extreme_index = analytical_min.index[0]

        analytical_first_new_x = self.analytical_data['x_axis'][first_index] + row['x_axis']
        analytical_last_new_x = self.analytical_data['x_axis'][last_index] + row['x_axis']
        analytical_extreme_new_x = self.analytical_data['x_axis'][extreme_index] + row['x_axis']

        new_x_r = np.linspace(analytical_extreme_new_x, analytical_last_new_x,
                              last_index - extreme_index + 1)
        new_x_l = np.linspace(analytical_first_new_x,
                              analytical_extreme_new_x - self.analytical_sampling_interval, extreme_index)
        new_x = np.concatenate((new_x_l, new_x_r))
        moved_analytical_data = self.analytical_data.copy()
        moved_analytical_data['x_axis'] = new_x
        # print(moved_analytical_data)

        return moved_analytical_data

    # method resamples data to base_data base
    def __resample_data(self):
        if self.superposed_analytical_data is None:
            self.move_and_superpose()

        if self.measured_data.shape[0] == self.superposed_analytical_data.shape[0]:
            return self.superposed_analytical_data

        x = self.superposed_analytical_data['x_axis']
        y = self.superposed_analytical_data['y_axis']
        f = interpolate.interp1d(x, y)

        first_x_val = self.measured_data['x_axis'].iloc[0]

        i = 1
        while x.iloc[0] > first_x_val:
            first_x_val = self.measured_data['x_axis'].iloc[i]
            i += 1

        last_x_val = self.measured_data['x_axis'].iloc[self.measured_data.shape[0] - 1]
        j = 1

        while x.iloc[-1] < last_x_val:
            last_x_val = self.measured_data['x_axis'].iloc[self.measured_data.shape[0] - (1 + j)]
            j += 1

        x_range = (last_x_val - first_x_val) / self.measured_data.shape[0]
        x_new = np.arange(first_x_val, last_x_val, x_range)
        y_new = f(x_new)

        resampling_product = pd.DataFrame(list(zip(self.measured_data['x_axis'], y_new)),
                                          columns=["x_axis", "y_axis"])
        return resampling_product
