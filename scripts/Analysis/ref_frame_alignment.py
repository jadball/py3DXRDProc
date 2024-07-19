import numpy as np

from py3DXRDProc.sample import Sample

sample_hdf5_path = "./nanox2_all_scans.hdf5"


sample = Sample.import_from_hdf5(sample_hdf5_path)


grains_1 = sample.get_load_step("Load_1").all_stitched_grains
grains_2 = sample.get_load_step("Load_2").all_stitched_grains
grains_3 = sample.get_load_step("Load_3").all_stitched_grains
grains_4 = sample.get_load_step("Load_4").all_stitched_grains
grains_5 = sample.get_load_step("Load_5").all_stitched_grains
grains_6 = sample.get_load_step("Load_6").all_stitched_grains
grains_7 = sample.get_load_step("Load_7").all_stitched_grains
grains_8 = sample.get_load_step("Load_8").all_stitched_grains
grains_9 = sample.get_load_step("Load_9").all_stitched_grains
grains_10 = sample.get_load_step("Load_10").all_stitched_grains


# guess initial affine transformations

# load_1 is well aligned
R1_guess = np.eye(3)
T1_guess = np.array([0, 0, 0])
D1_guess = np.zeros((4, 4))
D1_guess[0:3, 0:3] = R1_guess
D1_guess[0:3, 3] = T1_guess

# load_2 needs a shift in x
R2_guess = np.eye(3)
T2_guess = np.array([-0.04, 0, 0])
D2_guess = np.zeros((4, 4))
D2_guess[0:3, 0:3] = R2_guess
D2_guess[0:3, 3] = T2_guess

# load_3 needs a shift in x and y
R3_guess = np.eye(3)
T3_guess = np.array([-0.1, -0.05, 0])
D3_guess = np.zeros((4, 4))
D3_guess[0:3, 0:3] = R3_guess
D3_guess[0:3, 3] = T3_guess

# load_4 
R4_guess = np.eye(3)
T4_guess = np.array([-0.15, -0.05, 0])
D4_guess = np.zeros((4, 4))
D4_guess[0:3, 0:3] = R4_guess
D4_guess[0:3, 3] = T4_guess

# load_5 
R5_guess = np.eye(3)
T5_guess = np.array([-0.17, -0.05, 0])
D5_guess = np.zeros((4, 4))
D5_guess[0:3, 0:3] = R5_guess
D5_guess[0:3, 3] = T5_guess

# load_6
R6_guess = np.eye(3)
T6_guess = np.array([-0.18, -0.07, 0])
D6_guess = np.zeros((4, 4))
D6_guess[0:3, 0:3] = R6_guess
D6_guess[0:3, 3] = T6_guess

# load_7
R7_guess = np.eye(3)
T7_guess = np.array([-0.19, -0.07, 0])
D7_guess = np.zeros((4, 4))
D7_guess[0:3, 0:3] = R7_guess
D7_guess[0:3, 3] = T7_guess

# load_8
R8_guess = np.eye(3)
T8_guess = np.array([-0.2, -0.07, 0])
D8_guess = np.zeros((4, 4))
D8_guess[0:3, 0:3] = R8_guess
D8_guess[0:3, 3] = T8_guess

# load_9
R9_guess = np.eye(3)
T9_guess = np.array([-0.21, -0.08, 0])
D9_guess = np.zeros((4, 4))
D9_guess[0:3, 0:3] = R9_guess
D9_guess[0:3, 3] = T9_guess

# load_10
R10_guess = np.eye(3)
T10_guess = np.array([-0.15, -0.03, 0])
D10_guess = np.zeros((4, 4))
D10_guess[0:3, 0:3] = R10_guess
D10_guess[0:3, 3] = T10_guess

trans_dict = {
    "Load_1": D1_guess,
    "Load_2": D2_guess,
    "Load_3": D3_guess,
    "Load_4": D4_guess,
    "Load_5": D5_guess,
    "Load_6": D6_guess,
    "Load_7": D7_guess,
    "Load_8": D8_guess,
    "Load_9": D9_guess,
    "Load_10": D10_guess,
    
}


sample.guess_affine_transformations(trans_dict)


sample.optimize_sample_reference_frames("austenite")


sample.export_to_hdf5("./nanox2_all_scans_aligned.hdf5")