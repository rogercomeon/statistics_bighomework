import pandas as pd
import numpy as np
 
if __name__ == '__main__':
    unstrtf_lst = [[2.136, 1.778, 1.746, 2.565, 1.873, 2.413, 1.813, 1.72, 1.932, 1.987, 2.035, 2.178, 2.05, 2.016, 1.645, 1.756, 1.886, 2.106, 2.138, 1.914, 1.984, 1.906, 1.871, 1.939, 1.81, 1.93, 1.898, 1.802, 2.008, 1.724, 1.823, 1.636, 1.774, 2.055, 1.934, 1.629, 2.519, 2.093, 2.004, 1.793, 1.564, 1.962, 2.176, 1.846, 1.816, 2.018, 1.708, 2.465, 1.899, 1.523, 1.41, 2.102, 2.065, 2.402, 2.091, 1.867, 1.77, 1.466, 2.029, 1.659, 1.626, 1.977, 1.837, 2.13, 2.241, 2.184, 2.345, 1.833, 2.113, 1.764, 1.859, 1.868, 1.835, 1.906, 2.237, 1.846, 1.871, 1.769, 1.928, 1.831, 1.875, 2.039, 2.24, 1.835, 1.851]
    , [2.171, 1.831, 1.714, 2.507, 1.793, 2.526, 1.829, 1.705, 1.954, 2.017, 2.022, 2.16, 2.059, 1.966, 1.661, 1.752, 1.884, 2.203, 2.182, 1.97, 2.003, 1.875, 1.852, 1.884, 1.774, 1.916, 1.936, 1.809, 1.926, 1.717, 1.841, 1.59, 1.781, 2.016, 1.898, 1.657, 2.458, 2.134, 2.032, 1.785, 1.575, 1.959, 2.11, 1.854, 1.826, 1.992, 1.706, 2.419, 1.854, 1.514, 1.37, 2.084, 2.024, 2.398, 1.955, 1.859, 1.759, 1.441, 2.059, 1.653, 1.583, 1.987, 1.84, 2.106, 2.262, 2.13, 2.371, 1.776, 2.117, 1.733, 1.814, 1.839, 1.822, 1.883, 2.23, 1.803, 1.894, 1.783, 1.911, 1.813, 1.85, 2.004, 2.191, 1.823, 1.809]
    , [2.157, 1.873, 1.802, 2.761, 1.733, 2.506, 1.842, 1.765, 1.938, 2.058, 1.932, 2.196, 2.004, 2.126, 1.664, 1.698, 1.899, 2.073, 2.117, 2.083, 1.972, 1.969, 1.865, 1.937, 1.752, 1.939, 1.927, 1.804, 2.07, 1.725, 1.846, 1.5, 1.804, 2.1, 1.932, 1.773, 2.431, 2.088, 2.08, 1.812, 1.592, 1.953, 2.044, 2.019, 1.846, 2.061, 1.771, 2.254, 1.891, 1.536, 1.356, 1.952, 2.222, 2.427, 2.015, 1.873, 1.79, 1.384, 1.981, 1.665, 1.815, 2.006, 1.869, 2.102, 2.249, 2.27, 2.296, 1.814, 2.099, 1.702, 1.688, 1.89, 1.82, 1.927, 2.162, 1.825, 1.998, 1.811, 2.0, 1.842, 1.793, 2.115, 2.301, 1.789, 1.826]
    , [2.127, 1.744, 1.747, 2.548, 1.939, 2.296, 1.808, 1.71, 1.901, 1.906, 2.074, 2.167, 2.113, 2.044, 1.632, 1.821, 1.94, 2.076, 2.114, 1.837, 1.978, 1.904, 1.872, 1.98, 1.886, 1.923, 1.875, 1.799, 1.992, 1.704, 1.812, 1.715, 1.756, 2.061, 1.94, 1.554, 2.592, 2.065, 1.983, 1.802, 1.57, 1.955, 2.215, 1.765, 1.796, 2.006, 1.662, 2.573, 1.915, 1.543, 1.439, 2.16, 2.012, 2.42, 2.268, 1.886, 1.767, 1.527, 2.073, 1.65, 1.567, 2.016, 1.819, 2.153, 2.225, 2.237, 2.327, 1.877, 2.115, 1.804, 1.939, 1.867, 1.84, 1.905, 2.302, 1.883, 1.798, 1.725, 1.893, 1.846, 1.916, 2.025, 2.268, 1.867, 1.877]
    , [2.089, 1.664, 1.72, 2.441, 2.031, 2.321, 1.773, 1.702, 1.935, 1.968, 2.119, 2.191, 2.023, 1.925, 1.621, 1.75, 1.822, 2.074, 2.139, 1.764, 1.982, 1.873, 1.895, 1.955, 1.829, 1.945, 1.853, 1.794, 2.046, 1.75, 1.793, 1.741, 1.752, 2.042, 1.965, 1.532, 2.598, 2.086, 1.923, 1.771, 1.517, 1.98, 2.338, 1.743, 1.794, 2.014, 1.693, 2.618, 1.938, 1.5, 1.476, 2.216, 2.003, 2.361, 2.13, 1.85, 1.764, 1.513, 2.001, 1.669, 1.538, 1.897, 1.819, 2.163, 2.226, 2.099, 2.386, 1.865, 2.121, 1.818, 2.0, 1.876, 1.858, 1.908, 2.254, 1.874, 1.791, 1.759, 1.908, 1.822, 1.944, 2.012, 2.201, 1.863, 1.892]
    ]
 
    column_lst = ['whole_year', 'spring', 'summer', 'autumn', 'winter']
 
    # 计算列表两两间的相关系数
    data_dict = {} # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, unstrtf_lst):
        data_dict[col] = gf_lst
 
    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.corr() # 计算相关系数，得到一个矩阵
    print(cor1)
    print(unstrtf_df.columns.tolist())

