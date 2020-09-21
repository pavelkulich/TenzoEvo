DYNAMIC_SINGLE_WINKLER = [['EI', 3e5, 1e9],
                          ['m', 1e1, 1e4],
                          ['c', 1e4, 1e7],
                          ['k', 5e5, 5e9],
                          ['v', 1e1, 5e1]]

DYNAMIC_SINGLE_WINKLER_TEST = [['EI', 3e7],
                              ['m', 1e3], # kg/m
                              ['c', 1e5], # Nsm^(-2)
                              ['k', 5e7], # Nm^(-2)
                              ['v', 5e1], # ms^(-1)
                              ['Q', 2e5]] # N

DYNAMIC_DOUBLE_PASTERNAK = [['EI_1', 3e5, 3e7],
                            ['EI_2', 1, 1e4],
                            ['GA', 1e5, 1e6],
                            ['k_1', 1e6, 2e8],
                            ['k_2', 1e6, 2e8],
                            ['c_1', 5e4, 1e6],
                            ['c_2', 8e4, 1e6],
                            ['m_1', 1, 2e2],
                            ['m_2', 1e2, 1e3],
                            ['v', 1e1, 5e1]]

# DYNAMIC_DOUBLE_PASTERNAK = [['EI_1', 6e5, 7e6],
#                             ['EI_2', 1, 1e3],
#                             ['GA', 1e1, 1e7],
#                             ['k_1', 5e1, 5e6],
#                             ['k_2', 1e1, 2e5],
#                             ['c_1', 5e2, 3e5],
#                             ['c_2', 8e1, 1e6],
#                             ['m_1', 4e1, 9e1],
#                             ['m_2', 1, 1e3],
#                             ['v', 1e1, 4.5e1]]

DYNAMIC_DOUBLE_PASTERNAK_TEST = [['EI_1', 4500000],
                            ['EI_2', 1],
                            ['GA', 10000],
                            ['k_1', 250000000],
                            ['k_2', 40000000],
                            ['c_1', 90000],
                            ['c_2', 120000],
                            ['m_1', 55],
                            ['m_2', 157],
                            ['v', 20]]

Q = ['Q_', 4e5, 8e5]
