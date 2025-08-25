## --------- RG ----------------

# beta = 0.025
self.Q = np.array([[10.0, 0], [0, 10.0]])
self.R = np.array([[-5.125, 0], [0, -5.125]])
--- results ---
[INFO] [1747821880.964421965] [ref_gov_target]: Completion time: 67.8 
[INFO] [1747821880.964847283] [ref_gov_target]: Trajectory length: 4.827057207591496
[INFO] [1747821880.965197906] [ref_gov_target]: Tracking error: [0.01907]
[INFO] [1747824993.144395097] [ref_gov_target]: Min of Gamma: 0.0005048797580301212
[INFO] [1747824993.144763727] [ref_gov_target]: Max of Gamma: 0.00813089532973716


# beta = 0.5
self.Q = np.array([[10.0, 0], [0, 10.0]])
self.R = np.array([[-7.5003, 0], [0, -7.5003]]) # beta = 0.5
[INFO] [1747823031.904782633] [ref_gov_target]: Completion time: 47.2 
[INFO] [1747823031.905183165] [ref_gov_target]: Trajectory length: 4.834516910462213
[INFO] [1747823031.905561723] [ref_gov_target]: Tracking error: [0.01256]
[INFO] [1747824822.244461868] [ref_gov_target]: Min of Gamma: 0.000504561884313622
[INFO] [1747824822.244596261] [ref_gov_target]: Max of Gamma: 0.0075973385760081434

---- Choose this result
# beta = 1
self.Q = np.array([[10.0, 0], [0, 10.0]])
self.R = np.array([[-10, 0], [0, -10]])
--- result ---
[INFO] [1747821530.455145411] [ref_gov_target]: Completion time: 36.5 
[INFO] [1747821530.455623799] [ref_gov_target]: Trajectory length: 4.834115805235904
[INFO] [1747821530.456124368] [ref_gov_target]: Tracking error: [0.0093]
[INFO] [1747824647.846282156] [ref_gov_target]: Min of Gamma: 0.0005007260886076924
[INFO] [1747824647.846513000] [ref_gov_target]: Max of Gamma: 0.0042738448360781075
[INFO] [1747836116.924735223] [ref_gov_target]: Computation time offline: 0.009987354278564453


# beta = 2
self.Q = np.array([[10.0, 0], [0, 10.0]])
self.R = np.array([[-15.0, 0], [0, -15.0]]) # beta = 2.0
--- result ---
[INFO] [1747823199.617687655] [ref_gov_target]: Completion time: 29.400000000000002 
[INFO] [1747823199.618241316] [ref_gov_target]: Trajectory length: 4.823803036568332
[INFO] [1747823199.618819334] [ref_gov_target]: Tracking error: [0.0066]
[INFO] [1747824726.925946276] [ref_gov_target]: Min of Gamma: 0.00034213894512006985
[INFO] [1747824726.926076529] [ref_gov_target]: Max of Gamma: 0.001899486593812492

## -------- CBF --------
self.gamma = 1.0
self.omega = 0.1   # lyapunov inequality #0.05
self.relax_param = 0.1
---- result ----
[INFO] [1747833800.306140276] [cbf_mov_static_obs]: Completion time: 17.900000000000002 
[INFO] [1747833800.306417666] [cbf_mov_static_obs]: Trajectory length: 3.5059754133236383
[INFO] [1747833800.306718857] [cbf_mov_static_obs]: Tracking error: [0.0031]

[INFO] [1747835388.810600347] [cbf_mov_static_obs]: Completion time: 18.0 
[INFO] [1747835388.810791014] [cbf_mov_static_obs]: Trajectory length: 3.509010450756357
[INFO] [1747835388.811020235] [cbf_mov_static_obs]: Tracking error: [0.00198]
[INFO] [1747835388.811170618] [cbf_mov_static_obs]: Average computation time: 0.024158990383148192

---- Choose this result
[INFO] [1747835473.535406709] [cbf_mov_static_obs]: Completion time: 18.0 
[INFO] [1747835473.535652565] [cbf_mov_static_obs]: Trajectory length: 3.5101598122246385
[INFO] [1747835473.535962591] [cbf_mov_static_obs]: Tracking error: [0.00176]
[INFO] [1747835473.536262700] [cbf_mov_static_obs]: Average computation time: 0.024609560436672635

[INFO] [1747835604.468209530] [cbf_mov_static_obs]: Completion time: 17.900000000000002 
[INFO] [1747835604.468444583] [cbf_mov_static_obs]: Trajectory length: 3.507410230137102
[INFO] [1747835604.468679632] [cbf_mov_static_obs]: Tracking error: [0.00126]
[INFO] [1747835604.469139122] [cbf_mov_static_obs]: Average computation time: 0.02241738415297183
