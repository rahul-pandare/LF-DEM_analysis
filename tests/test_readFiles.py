import src.readFiles as readFiles # type: ignore
import pytest                     # type: ignore
from   pathlib import Path
import glob
import random
import numpy as np                # type: ignore

'''
Jul 22, 2025, RVP - Initial file
Oct 06, 2025, RVP - added 'src.readFile' making script independent of
                    the location of readFiles.

This file contains unit tests for the readFiles module.
It tests the functions that read and process data files, ensuring they return the expected structures and values
'''

# Path to test data directory
# Update this path to the directory where your test data files are located
test_dir = Path('./test_data').resolve()

# --------------------------------------------
# Tests for reading rigid cluster IDs
# --------------------------------------------

def test_rigClusterList():
    expected_data = [
        [174, 201, 174, 488, 201, 488],
        [206, 767, 206, 904, 767, 904],
        [318, 603, 318, 947, 603, 947],
        [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
        [206, 767, 206, 904, 767, 904],
        [41, 546, 41, 772, 546, 772],
        [83, 816, 83, 895, 816, 895],
        [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
        [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
        [323, 388, 323, 875, 388, 875],
        [337, 745, 337, 982, 745, 982],
        [434, 668, 434, 768, 668, 768],
        [41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772],
        [69, 226, 69, 897, 226, 897],
        [83, 816, 83, 895, 816, 895],
        [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
        [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
        [323, 388, 323, 875, 388, 875],
        [327, 354, 327, 774, 354, 774],
        [434, 668, 434, 768, 668, 768],
        [41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772],
        [64, 133, 64, 635, 133, 453, 133, 909, 453, 909, 635, 909],
        [69, 226, 69, 897, 226, 897],
        [83, 816, 83, 895, 816, 895],
        [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
        [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947]
    ]

    rigFile = list(test_dir.glob('rig_*test.dat'))
    assert rigFile, f"No test file matching 'rig_*test.dat' found in {test_dir}"

    file_path = rigFile[0]
    with open(file_path, 'r') as f:
        rigList = readFiles.rigClusterList(f)

    # Type and structure checks
    assert isinstance(rigList, list), "Returned object is not a list"
    assert all(isinstance(cluster, list) for cluster in rigList), "Each cluster should be a list"
    assert all(isinstance(val, int) for cluster in rigList for val in cluster), "All elements should be integers"
    
    # Size check
    assert len(rigList) == len(expected_data), f"Expected {len(expected_data)} clusters, got {len(rigList)}"

    # Content check
    assert rigList == expected_data, "Cluster data does not match expected output"


def test_rigList():
    expected_data = [[[0]],
            [[0]],
            [[174, 201, 174, 488, 201, 488],
            [206, 767, 206, 904, 767, 904],
            [318, 603, 318, 947, 603, 947]],
            [[135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
            [206, 767, 206, 904, 767, 904]],
            [[41, 546, 41, 772, 546, 772],
            [83, 816, 83, 895, 816, 895],
            [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
            [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
            [323, 388, 323, 875, 388, 875],
            [337, 745, 337, 982, 745, 982],
            [434, 668, 434, 768, 668, 768]],
            [[41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772],
            [69, 226, 69, 897, 226, 897],
            [83, 816, 83, 895, 816, 895],
            [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
            [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947],
            [323, 388, 323, 875, 388, 875],
            [327, 354, 327, 774, 354, 774],
            [434, 668, 434, 768, 668, 768]],
            [[41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772],
            [64, 133, 64, 635, 133, 453, 133, 909, 453, 909, 635, 909],
            [69, 226, 69, 897, 226, 897],
            [83, 816, 83, 895, 816, 895],
            [84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975],
            [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947]]]

    rigFile = list(test_dir.glob('rig_*test.dat'))
    assert rigFile, f"No file matching 'rig_*test.dat' found in {test_dir}"

    file_path = rigFile[0]

    # Read actual rig list using context manager
    with open(file_path, 'r') as f:
        actual_data = readFiles.rigList(f)

    # Structural checks
    assert isinstance(actual_data, list), "Returned object is not a list (outermost level)"
    assert all(isinstance(step, list) for step in actual_data), "Each timestep should be a list"
    assert all(isinstance(cluster, list) for step in actual_data for cluster in step), "Each cluster should be a list"
    assert all(isinstance(val, int) for step in actual_data for cluster in step for val in cluster), "All elements must be integers"

    # Length check
    assert len(actual_data) == len(expected_data), f"Expected {len(expected_data)} timesteps, got {len(actual_data)}"

    # Content check
    assert actual_data == expected_data, "rigList content does not match expected structure and values"

def test_rigListFlat():
    expected_data = [[0], [0],
            [174, 201, 174, 488, 201, 488, 206, 767, 206, 904, 767, 904, 318, 603, 318, 947, 603, 947],
            
            [135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947,
             206, 767, 206, 904, 767, 904],
            
            [41, 546, 41, 772, 546, 772,
            83, 816, 83, 895, 816, 895,
            84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975,
            135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947,
            323, 388, 323, 875, 388, 875,
            337, 745, 337, 982, 745, 982,
            434, 668, 434, 768, 668, 768],
            
            [41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772,
            69, 226, 69, 897, 226, 897,
            83, 816, 83, 895, 816, 895,
            84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975,
            135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947,
            323, 388, 323, 875, 388, 875,
            327, 354, 327, 774, 354, 774,
            434, 668, 434, 768, 668, 768],
            
            [41, 546, 41, 772, 401, 584, 401, 772, 546, 772, 584, 772,
            64, 133, 64, 635, 133, 453, 133, 909, 453, 909, 635, 909,
            69, 226, 69, 897, 226, 897,
            83, 816, 83, 895, 816, 895,
            84, 929, 84, 933, 84, 975, 190, 929, 190, 975, 933, 975,
            135, 603, 135, 646, 318, 603, 318, 947, 603, 947, 646, 947]]

    rigFile = list(test_dir.glob('rig_*test.dat'))
    assert rigFile, f"No file matching 'rig_*test.dat' found in {test_dir}"

    file_path = rigFile[0]

    # Read actual rig list using context manager
    with open(file_path, 'r') as f:
        actual_data = readFiles.rigListFlat(f)

    # Structural checks
    assert isinstance(actual_data, list), "Returned object is not a list (outermost level)"
    assert all(isinstance(step, list) for step in actual_data), "Each timestep should be a list"
    #assert all(isinstance(cluster, list) for step in actual_data for cluster in step), "Each cluster should be a list"
    assert all(isinstance(val, int) for step in actual_data for val in step), "All elements must be integers"

    # Length check
    assert len(actual_data) == len(expected_data), f"Expected {len(expected_data)} timesteps, got {len(actual_data)}"

    # Content check
    assert actual_data == expected_data, "rigListFlat content does not match expected structure and values"
    
# --------------------------------------------
# Tests for reading particle size list
# --------------------------------------------

def test_particleSizeList():
    # Expected particle sizes
    expected = np.array([1.0, 1.0, 1.0, 1.4, 1.4, 1.4])

    ranSeed_file = list(test_dir.glob('random_seed_test.dat'))
    assert ranSeed_file, f"No file matching 'random_seed_test.dat' found in {test_dir}"

    file_path = ranSeed_file[0]
    with open(file_path, 'r') as f:
        particle_sizes = readFiles.particleSizeList(f, sizeRatio=1.4)

    # Type checks
    assert isinstance(particle_sizes, np.ndarray), "Returned object is not a NumPy array"
    assert particle_sizes.dtype == np.float64, f"Expected dtype float64, got {particle_sizes.dtype}"

    # Shape check
    assert particle_sizes.shape == expected.shape, f"Expected shape {expected.shape}, got {particle_sizes.shape}"

    # Value check
    assert np.array_equal(particle_sizes, expected), "Particle sizes do not match expected values"
    

def test_particleSizeList_mono():
    '''
    Testing specifically the monodisperse case where sizeRatio=1
    '''
    random.seed(42) # Setting seed for reproducibility

    # Generate synthetic particle size list (no file input, monodisperse size ratio)
    particle_sizes = readFiles.particleSizeList(None, sizeRatio=1, npp=6)

    expected = [2, 1, 1, 2, 1, 2] # Expected result based on random.seed(42) — deterministic output

    # Type checks
    assert isinstance(particle_sizes, list), "Returned object is not a list"
    assert all(isinstance(x, int) for x in particle_sizes), "All particle sizes must be integers"

    # Size check
    assert len(particle_sizes) == len(expected), f"Expected {len(expected)} elements, got {len(particle_sizes)}"

    # Value check
    assert particle_sizes == expected, f"Particle sizes do not match expected: {particle_sizes} != {expected}"

# --------------------------------------------
# Test for reading interaction file
# --------------------------------------------

def test_interactionsList():
    expected_data = [np.array([
    [0, 891, 0.429056, 0, -0.903278, 0.0179601, 61.1331, -5.30821, 0, -2.5214, 0, 0, 0, 0, 0, 0, 0.379068],
    [0, 928, -0.99308, 0, 0.117444, 0.0173358, 32.701, -1.05412, 0, -8.91341, 0, 0, 0, 0, 0, 0, 0.38479],
    [0, 277, 0.305526, 0, 0.952184, 0.0178282, -12.766, 2.81001, 0, -0.901644, 0, 0, 0, 0, 0, 0, 0.350039],
    [0, 790, 0.992921, 0, 0.118778, 0.018714, -7.53486, -0.346116, 0, 2.89334, 0, 0, 0, 0, 0, 0, 0.372271],
    [1, 51, -0.849228, 0, -0.528026, 0.370001, -7.69429, 0.277562, 0, -0.446406, 0, 0, 0, 0, 0, 0, 0],
    [1, 997, -0.875928, 0, 0.482442, 0.0195719, 49.9058, -2.22647, 0, -4.04242, 0, 0, 0, 0, 0, 0, 0.364685],
    [1, 263, -0.0882054, 0, -0.996102, 0.124799, -12.9726, -4.15579, 0, 0.367998, 0, 0, 0, 0, 0, 0, 0.0412079],
    [1, 373, 0.998801, 0, -0.0489579, 0.117457, 7.68127, 0.0891717, 0, 1.81921, 0, 0, 0, 0, 0, 0, 0.0477254],
    [1, 882, 0.480419, 0, 0.877039, 0.0197062, -63.2718, 2.11505, 0, -1.15857, 0, 0, 0, 0, 0, 0, 0.363511],
    [2, 171, -0.996066, 0, 0.0886113, 0.0191557, -1.20976, -0.574336, 0, -6.45602, 0, 0, 0, 0, 0, 0, 0.340868],
    [2, 548, -0.0981543, 0, 0.995171, 0.019963, 1.06953, 7.92073, 0, 0.781226, 0, 0, 0, 0, 0, 0, 0.335408],
    [2, 519, 0.518431, 0, -0.85512, 0.0196108, 29.8601, -2.8973, 0, -1.75654, 0, 0, 0, 0, 0, 0, 0.337779]]),
    np.array([
    [0, 891, -0.387686, 0, -0.921792, -0.000620774, 0, 0, 0, 0, 2, 17.3855, 3.34303, 0, -1.40601, 0, 0.583333],
    [544, 987, -0.965542, 0, 0.260247, -0.00100389, 0, 0, 0, 0, 2, 28.7741, -2.50531, 0, -9.29495, 0, 0.583333],
    [140, 663, -0.0421651, 0, 0.999111, -0.00034203, 0, 0, 0, 0, 2, 9.45874, -10.9847, 0, -0.463583, 0, 0.583333],
    [287, 917, -0.0292224, 0, 0.999573, 0.138554, 2.31485, -0.651975, 0, -0.0190604, 0, 0, 0, 0, 0, 0, 0.0209775],
    [637, 978, 0.81575, 0, -0.578405, -0.00190128, 0, 0, 0, 0, 2, 51.6871, 3.81939, 0, 5.38665, 0, 0.583333],
    [1, 997, -0.12234, 0, 0.992488, -0.00084553, 0, 0, 0, 0, 2, 24.0285, -26.5413, 0, -3.27165, 0, 0.583333],
    [701, 791, 0.24242, 0, -0.970171, -0.00334231, 0, 0, 0, 0, 2, 131.81, -43.4331, 0, -10.8528, 0, 0.7],
    [1, 373, 0.796499, 0, -0.604639, 0.16974, -0.692444, 0.0294166, 0, 0.0387508, 0, 0, 0, 0, 0, 0, 0.0167736],
    [1, 882, 0.900114, 0, 0.435655, 0.00442566, -7.62161, 0.129789, 0, -0.268159, 0, 0, 0, 0, 0, 0, 0.524551],
    [168, 368, -0.259167, 0, 0.965833, 0.344576, 0.329678, 0.0909774, 0, 0.0244124, 0, 0, 0, 0, 0, 0, 0.000508182],
    [2, 548, 0.59232, 0, 0.805703, 0.00724076, -1.30006, 0.889782, 0, -0.654132, 0, 0, 0, 0, 0, 0, 0.432591],
    [287, 506, -0.490229, 0, -0.871594, 0.312347, -0.0544877, 0.165343, 0, -0.0929973, 0, 0, 0, 0, 0, 0, 0.000968176],
    [931, 957, 0.488719, 0, 0.872442, -0.00113329, 0, 0, 0, 0, 2, 44.3011, -5.21637, 0, 2.92207, 0, 0.7],
    [797, 927, -0.957037, 0, 0.289967, -0.000580239, 0, 0, 0, 0, 2, 22.6114, 5.77074, 0, 19.0463, 0, 0.7],
    [3, 547, 0.92241, 0, 0.386213, -0.000599011, 0, 0, 0, 0, 2, 11.1343, 6.74475, 0, -16.1088, 0, 0.5],
    [3, 590, -0.611653, 0, -0.791126, 0.38338, -0.217735, -0.00604272, 0, 0.00467188, 0, 0, 0, 0, 0, 0, 0]]),   
    np.array([
    [0, 891, -0.391516, 0, -0.920171, -0.000138442, 0, 0, 0, 0, 2, 3.12589, 7.46932, 0, -3.17806, 0, 0.583333],
    [544, 987, -0.967276, 0, 0.253727, -0.00127237, 0, 0, 0, 0, 2, 37.0393, -2.9697, 0, -11.3213, 0, 0.583333],
    [140, 663, -0.042719, 0, 0.999087, -3.73364e-05, 0, 0, 0, 0, 2, 1.12575, -1.66239, 0, -0.0710807, 0, 0.583333]])]
    
    intFile  = list(test_dir.glob('int_*test.dat'))
    assert intFile, f"No file matching 'int_*test.dat' found in {test_dir}"

    file_path = intFile[0]

    # Read the interactions list safely
    with open(file_path, 'r') as f:
        int_list = readFiles.interactionsList(f)

    # Basic type check for the outer structure
    assert isinstance(int_list, list), "interactionsList should return a list"

    # Check that the length matches expected
    assert len(int_list) == len(expected_data), f"Expected {len(expected_data)} arrays, got {len(int_list)}"

    # Compare each numpy array with expected
    for idx, (actual_array, expected_array) in enumerate(zip(int_list, expected_data)):
        assert isinstance(actual_array, np.ndarray), f"Element {idx} is not a numpy array"
        assert actual_array.shape == expected_array.shape, (
            f"Shape mismatch at element {idx}: expected {expected_array.shape}, got {actual_array.shape}"
        )
        np.testing.assert_array_equal(actual_array, expected_array, err_msg=f"Mismatch at element {idx}")

# --------------------------------------------
# Tests for reading particle file
# --------------------------------------------

def get_expected_par_data():
    '''
    Returns the expected data for the particle file test.
    This is used to compare against the output of readFiles.readParFile* function.
    '''
    return [np.array([
           [0, 1, -6.94832, -0.287701, 0.421004, 0, 2.76226, 0, 1.8458, 0, 0],
           [1, 1, 15.9122, -5.82255, -16.6717, 0, -0.256502, 0, 1.83494, 0, 0],
           [2, 1, 36.0294, 11.9706, 44.6729, 0, 1.01069, 0, 1.83261, 0, 0],
           [3, 1, -15.2673, -0.71701, 0.384757, 0, -0.597229, 0, 1.54845, 0, 0],
           [4, 1, -26.6757, 17.6198, 67.996, 0, 2.47818, 0, 1.26766, 0, 0],
           [5, 1, -30.6189, 28.9987, 112.927, 0, 4.4265, 0, 1.56611, 0, 0]]),
           np.array([
           [0, 1, -8.13108, -1.20098, -1.12063, 0, -1.59272, 0, 0.775525, 0, 1.19039],
           [1, 1, 9.90338, -6.83313, -2.01544, 0, 0.0829431, 0, 0.227265, 0, 0.736263],
           [2, 1, -25.4824, 11.6838, 3.89886, 0, 0.0024977, 0, 0.310855, 0, -0.272455],
           [3, 1, -18.5279, -2.45106, -0.478714, 0, 0.63809, 0, -0.580693, 0, 0.363456],
           [4, 1, -8.60271, 20.3819, 4.67049, 0, 1.3447, 0, 1.69814, 0, 0.368413],
           [5, 1, -4.16724, 31.294, 7.20103, 0, -0.0753068, 0, -2.25045, 0, -0.268695]]),
           np.array([
           [0, 1, -8.21288, -1.26632, -1.46194, 0, -1.40088, 0, 0.976844, 0, 1.22964],
           [1, 1, 9.80855, -6.82294, -2.19529, 0, 0.350074, 0, 0.288802, 0, 0.746443],
           [2, 1, -25.3285, 11.6786, 3.27815, 0, -0.22184, 0, 0.30429, 0, -0.259056],
           [3, 1, -18.5538, -2.43262, -1.03997, 0, -0.53329, 0, -0.0792144, 0, 0.341145],
           [4, 1, -8.40078, 20.4337, 4.34022, 0, 0.872488, 0, 1.27026, 0, 0.435025],
           [5, 1, -3.88117, 31.2704, 5.61757, 0, -1.56217, 0, 1.08328, 0, -0.320634]])]
    
@pytest.mark.parametrize("read_func", 
    [readFiles.readParFile, readFiles.readParFile2, readFiles.readParFile3, readFiles.readParFile4],
    ids=["readParFile", "readParFile2", "readParFile3", "readParFile4"])
def test_readParFile_all(read_func):
   
    expected = get_expected_par_data()  # Get expected data
    parFile  = glob.glob(f'{test_dir}/par_*test.dat')
    assert parFile, "Test file not found"
    
    file_path = parFile[0]

    # Call function with appropriate arguments
    if read_func.__name__ == 'readParFile':
        result = read_func(open(file_path))
    elif read_func.__name__ == 'readParFile2':
        result = read_func(file_path)
    elif read_func.__name__ == 'readParFile3':
        result = read_func(open(file_path), final_strain=0.02)
    elif read_func.__name__ == 'readParFile4':
        result = read_func(open(file_path), npp=6)
    else:
        raise ValueError(f"Unknown function: {read_func.__name__}")

    # Assert type
    assert isinstance(result, list), f"{read_func.__name__} did not return a list"
    assert all(isinstance(arr, np.ndarray) for arr in result), f"{read_func.__name__} list items are not numpy arrays"

    # Assert size
    assert len(result) == len(expected), f"{read_func.__name__} returned {len(result)} arrays, expected {len(expected)}"

    # Assert values
    for i, (a, b) in enumerate(zip(result, expected)):
        np.testing.assert_array_equal(a, b, err_msg=f"Mismatch in array index {i} from {read_func.__name__}")

# --------------------------------------------
# Test for freeing memory
# --------------------------------------------

def test_free_mem_zeros_arrays():
    # Create arrays with non-zero data
    array1 = np.array([1, 2, 3, 4], dtype=np.int32)
    array2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    
    # Call free_mem
    readFiles.free_mem(array1, array2)
    
    # Assert all elements are zero after calling free_mem
    assert np.all(array1 == 0), "arr1 was not zeroed out"
    assert np.all(array2 == 0), "arr2 was not zeroed out"