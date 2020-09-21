import numpy as np
import pandas as pd

models = [
    'dynamic_double_pasternak',
    'dynamic_single_winkler',
    'dynamic_double_winkler',
    'dynamic_single_pasternak'
]


class Model:
    def __init__(self, type, moment=True, x_interval=30):
        self.x_interval = x_interval
        self.moment = moment
        if type not in models:
            self.type = 'dynamic_single_winkler'
        else:
            self.type = type

    def get_model_type(self):
        return self.type

    def calculate_model(self, params, q):
        if self.type == 'dynamic_double_pasternak':
            return self.dynamic_double_pasternak(params, q)
        elif self.type == 'dynamic_single_winkler':
            return self.dynamic_single_winkler(params, q)
        elif self.type == 'dynamic_double_winkler':
            pass
        elif self.type == 'dynamic_single_pasternak':
            pass
        else:
            return None

    def dynamic_single_winkler(self, params, q):
        ei = params['EI']
        m = params['m']
        c = params['c']
        k = params['k']
        v = params['v']

        lmbd = np.power((k / (4 * ei)), 0.25)
        char_len = 1 / lmbd

        dyn_load = q / (ei * lmbd ** 3)

        alpha = (v / (2 * lmbd) * np.sqrt(m / ei))
        beta = (c / (2 * m) * np.sqrt(m / k))

        rts = np.roots([1, 0, 4 * alpha ** 2, 8 * alpha * beta, 4])
        sorted_rts = np.sort(rts)

        matrix_g = np.array([
            [1, 1, -1, -1],
            [sorted_rts[0], sorted_rts[1], -sorted_rts[2], -sorted_rts[3]],
            [sorted_rts[0] ** 2, sorted_rts[1] ** 2, -sorted_rts[2] ** 2, -sorted_rts[3] ** 2],
            [sorted_rts[0] ** 3, sorted_rts[1] ** 3, -sorted_rts[2] ** 3, -sorted_rts[3] ** 3]
        ])

        inv_matrix_g = np.linalg.inv(matrix_g)
        p = np.array([0, 0, 0, dyn_load])
        p_trans = p.transpose()

        consts_a = np.matmul(inv_matrix_g, p_trans)

        s_r = np.linspace(0, self.x_interval, 500)
        s_l = np.linspace(-self.x_interval, 0, 500)

        w_r = consts_a[0] * np.exp(sorted_rts[0] * s_r) + consts_a[1] * np.exp(sorted_rts[1] * s_r)
        w_l = consts_a[2] * np.exp(sorted_rts[2] * s_l) + consts_a[3] * np.exp(sorted_rts[3] * s_l)

        defl = np.concatenate((w_l[:-1], w_r))
        x_axis = np.concatenate((s_l[:-1], s_r)) * char_len / v

        if self.moment:
            diff_x = np.diff(x_axis)
            first_diff = np.diff(defl) / diff_x
            second_diff = np.diff(first_diff) / diff_x[1:]
            return df_from_lists(x_axis, np.real(second_diff * 1000000))

        return df_from_lists(x_axis, -np.real(defl) * 10000)

    def dynamic_double_pasternak(self, params, q):
        ei1 = params['EI_1']
        ei2 = params['EI_2']
        ga = params['GA']
        k1 = params['k_1']
        k2 = params['k_2']
        c1 = params['c_1']
        c2 = params['c_2']
        m1 = params['m_1']
        m2 = params['m_2']
        v = params['v']

        s_r = np.linspace(0, self.x_interval, 500)
        s_l = np.linspace(-self.x_interval, 0, 500)

        lmbd = (k1 / (4 * ei1)) ** (1 / 4)
        char_len = 1 / lmbd

        # konstanty diferencialni rovnice
        con_a = ei2
        con_b = 0
        con_c = (ei1 * m2 * v ** 2 + ei2 * m1 * v ** 2 - ga * ei1) / (ei1 * lmbd ** 2)
        con_d = -(v * (ei1 * c2 + ei2 * c1)) / (ei1 * lmbd ** 3)
        con_e = (4 * (ei1 * k1 + ei1 * k2 + ei2 * k1 + m1 * m2 * v ** 4 - ga * m1 * v ** 2)) / k1
        con_f = -(4 * v * (c1 * m2 * v ** 2 + c2 * m1 * v ** 2 - ga * c1)) / (k1 * lmbd)
        con_g = (4 * (m1 * k1 * v ** 2 + m1 * k2 * v ** 2 + m2 * k1 * v ** 2 + c1 * c2 * v ** 2 - ga * k1)) / (
                k1 * lmbd ** 2)
        con_h = -((4 * v) * (c1 * k1 + c1 * k2 + c2 * k1)) / (k1 * lmbd ** 3)
        con_i = (16 * ei1 * k2) / k1

        eq_members = [con_a, con_b, con_c, con_d, con_e, con_f, con_g, con_h, con_i]
        eq_roots = np.roots(eq_members)

        sorted_eq_roots = np.sort(eq_roots)

        # konstanty vychazejici z rovnice 4.19
        kon1 = (ei1 * lmbd ** 4) / k1
        kon2 = (m1 * v ** 2 * lmbd ** 2) / (k1)
        kon3 = -(c1 * v * lmbd) / (k1)

        # vypocet clenu matice G
        l5r1 = kon1 * sorted_eq_roots[0] ** 4 + kon2 * sorted_eq_roots[0] ** 2 - kon3 * sorted_eq_roots[0] + 1
        l5r2 = kon1 * sorted_eq_roots[1] ** 4 + kon2 * sorted_eq_roots[1] ** 2 - kon3 * sorted_eq_roots[1] + 1
        l5r3 = kon1 * sorted_eq_roots[2] ** 4 + kon2 * sorted_eq_roots[2] ** 2 - kon3 * sorted_eq_roots[2] + 1
        l5r4 = kon1 * sorted_eq_roots[3] ** 4 + kon2 * sorted_eq_roots[3] ** 2 - kon3 * sorted_eq_roots[3] + 1
        l5r5 = -kon1 * sorted_eq_roots[4] ** 4 - kon2 * sorted_eq_roots[4] ** 2 + kon3 * sorted_eq_roots[4] - 1
        l5r6 = -kon1 * sorted_eq_roots[5] ** 4 - kon2 * sorted_eq_roots[5] ** 2 + kon3 * sorted_eq_roots[5] - 1
        l5r7 = -kon1 * sorted_eq_roots[6] ** 4 - kon2 * sorted_eq_roots[6] ** 2 + kon3 * sorted_eq_roots[6] - 1
        l5r8 = -kon1 * sorted_eq_roots[7] ** 4 - kon2 * sorted_eq_roots[7] ** 2 + kon3 * sorted_eq_roots[7] - 1
        l6r1 = kon1 * sorted_eq_roots[0] ** 5 + kon2 * sorted_eq_roots[0] ** 3 - kon3 * sorted_eq_roots[0] ** 2 + \
               sorted_eq_roots[0]
        l6r2 = kon1 * sorted_eq_roots[1] ** 5 + kon2 * sorted_eq_roots[1] ** 3 - kon3 * sorted_eq_roots[1] ** 2 + \
               sorted_eq_roots[1]
        l6r3 = kon1 * sorted_eq_roots[2] ** 5 + kon2 * sorted_eq_roots[2] ** 3 - kon3 * sorted_eq_roots[2] ** 2 + \
               sorted_eq_roots[2]
        l6r4 = kon1 * sorted_eq_roots[3] ** 5 + kon2 * sorted_eq_roots[3] ** 3 - kon3 * sorted_eq_roots[3] ** 2 + \
               sorted_eq_roots[3]
        l6r5 = -kon1 * sorted_eq_roots[4] ** 5 - kon2 * sorted_eq_roots[4] ** 3 + kon3 * sorted_eq_roots[4] ** 2 - \
               sorted_eq_roots[4]
        l6r6 = -kon1 * sorted_eq_roots[5] ** 5 - kon2 * sorted_eq_roots[5] ** 3 + kon3 * sorted_eq_roots[5] ** 2 - \
               sorted_eq_roots[5]
        l6r7 = -kon1 * sorted_eq_roots[6] ** 5 - kon2 * sorted_eq_roots[6] ** 3 + kon3 * sorted_eq_roots[6] ** 2 - \
               sorted_eq_roots[6]
        l6r8 = -kon1 * sorted_eq_roots[7] ** 5 - kon2 * sorted_eq_roots[7] ** 3 + kon3 * sorted_eq_roots[7] ** 2 - \
               sorted_eq_roots[7]
        l7r1 = kon1 * sorted_eq_roots[0] ** 6 + kon2 * sorted_eq_roots[0] ** 4 - kon3 * sorted_eq_roots[0] ** 3 + \
               sorted_eq_roots[0] ** 2
        l7r2 = kon1 * sorted_eq_roots[1] ** 6 + kon2 * sorted_eq_roots[1] ** 4 - kon3 * sorted_eq_roots[1] ** 3 + \
               sorted_eq_roots[1] ** 2
        l7r3 = kon1 * sorted_eq_roots[2] ** 6 + kon2 * sorted_eq_roots[2] ** 4 - kon3 * sorted_eq_roots[2] ** 3 + \
               sorted_eq_roots[2] ** 2
        l7r4 = kon1 * sorted_eq_roots[3] ** 6 + kon2 * sorted_eq_roots[3] ** 4 - kon3 * sorted_eq_roots[3] ** 3 + \
               sorted_eq_roots[3] ** 2
        l7r5 = -kon1 * sorted_eq_roots[4] ** 6 - kon2 * sorted_eq_roots[4] ** 4 + kon3 * sorted_eq_roots[4] ** 3 - \
               sorted_eq_roots[4] ** 2
        l7r6 = -kon1 * sorted_eq_roots[5] ** 6 - kon2 * sorted_eq_roots[5] ** 4 + kon3 * sorted_eq_roots[5] ** 3 - \
               sorted_eq_roots[5] ** 2
        l7r7 = -kon1 * sorted_eq_roots[6] ** 6 - kon2 * sorted_eq_roots[6] ** 4 + kon3 * sorted_eq_roots[6] ** 3 - \
               sorted_eq_roots[6] ** 2
        l7r8 = -kon1 * sorted_eq_roots[7] ** 6 - kon2 * sorted_eq_roots[7] ** 4 + kon3 * sorted_eq_roots[7] ** 3 - \
               sorted_eq_roots[7] ** 2
        l8r1 = kon1 * sorted_eq_roots[0] ** 7 + kon2 * sorted_eq_roots[0] ** 5 - kon3 * sorted_eq_roots[0] ** 4 + \
               sorted_eq_roots[0] ** 3
        l8r2 = kon1 * sorted_eq_roots[1] ** 7 + kon2 * sorted_eq_roots[1] ** 5 - kon3 * sorted_eq_roots[1] ** 4 + \
               sorted_eq_roots[1] ** 3
        l8r3 = kon1 * sorted_eq_roots[2] ** 7 + kon2 * sorted_eq_roots[2] ** 5 - kon3 * sorted_eq_roots[2] ** 4 + \
               sorted_eq_roots[2] ** 3
        l8r4 = kon1 * sorted_eq_roots[3] ** 7 + kon2 * sorted_eq_roots[3] ** 5 - kon3 * sorted_eq_roots[3] ** 4 + \
               sorted_eq_roots[3] ** 3
        l8r5 = -kon1 * sorted_eq_roots[4] ** 7 - kon2 * sorted_eq_roots[4] ** 5 + kon3 * sorted_eq_roots[4] ** 4 - \
               sorted_eq_roots[4] ** 3
        l8r6 = -kon1 * sorted_eq_roots[5] ** 7 - kon2 * sorted_eq_roots[5] ** 5 + kon3 * sorted_eq_roots[5] ** 4 - \
               sorted_eq_roots[5] ** 3
        l8r7 = -kon1 * sorted_eq_roots[6] ** 7 - kon2 * sorted_eq_roots[6] ** 5 + kon3 * sorted_eq_roots[6] ** 4 - \
               sorted_eq_roots[6] ** 3
        l8r8 = -kon1 * sorted_eq_roots[7] ** 7 - kon2 * sorted_eq_roots[7] ** 5 + kon3 * sorted_eq_roots[7] ** 4 - \
               sorted_eq_roots[7] ** 3

        matrix_g = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
                             [-sorted_eq_roots[0], -sorted_eq_roots[1], -sorted_eq_roots[2], -sorted_eq_roots[3],
                              sorted_eq_roots[4], sorted_eq_roots[5], sorted_eq_roots[6], sorted_eq_roots[7]],
                             [-sorted_eq_roots[0] ** 2, -sorted_eq_roots[1] ** 2, -sorted_eq_roots[2] ** 2,
                              -sorted_eq_roots[3] ** 2, sorted_eq_roots[4] ** 2, sorted_eq_roots[5] ** 2,
                              sorted_eq_roots[6] ** 2, sorted_eq_roots[7] ** 2],
                             [-sorted_eq_roots[0] ** 3, -sorted_eq_roots[1] ** 3, -sorted_eq_roots[2] ** 3,
                              -sorted_eq_roots[3] ** 3, sorted_eq_roots[4] ** 3, sorted_eq_roots[5] ** 3,
                              sorted_eq_roots[6] ** 3, sorted_eq_roots[7] ** 3],
                             [-l5r1, -l5r2, -l5r3, -l5r4, -l5r5, -l5r6, -l5r7, -l5r8],
                             [-l6r1, -l6r2, -l6r3, -l6r4, -l6r5, -l6r6, -l6r7, -l6r8],
                             [-l7r1, -l7r2, -l7r3, -l7r4, -l7r5, -l7r6, -l7r7, -l7r8],
                             [-l8r1, -l8r2, -l8r3, -l8r4, -l8r5, -l8r6, -l8r7, -l8r8]])

        p = np.array([0, 0, 0, q / (ei1 * lmbd ** 3), 0, 0, 0, 0])
        p_trans = p.transpose()

        inv_matrix_g = np.linalg.inv(matrix_g)

        const_a = np.matmul(inv_matrix_g, p_trans)

        # svisle posunuti vpravo od osy y
        w1r = const_a[0] * np.exp(sorted_eq_roots[0] * s_r) + const_a[1] * np.exp(sorted_eq_roots[1] * s_r) + const_a[
            2] * np.exp(sorted_eq_roots[2] * s_r) + const_a[3] * np.exp(
            sorted_eq_roots[3] * s_r);

        w1l = const_a[4] * np.exp(sorted_eq_roots[4] * s_l) + const_a[5] * np.exp(sorted_eq_roots[5] * s_l) + const_a[
            6] * np.exp(sorted_eq_roots[6] * s_l) + const_a[7] * np.exp(sorted_eq_roots[7] * s_l);

        w1 = np.concatenate((w1l, w1r[1::]))
        # x_axis = np.concatenate((s_l, s_r))
        x_axis = np.concatenate((s_l[:-1], s_r)) * char_len / v

        if self.moment:
            diff_x = np.diff(x_axis)
            first_diff = np.diff(w1)/diff_x
            second_diff = np.diff(first_diff)/diff_x[1:]
            return df_from_lists(x_axis, -np.real(second_diff * 10000))

        return df_from_lists(x_axis, np.real(w1))


def df_from_lists(x_axis, y_axis):
    df = pd.DataFrame(list(zip(x_axis, y_axis)), columns=["x_axis", "y_axis"])
    return df
