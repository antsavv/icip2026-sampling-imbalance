from plyfile import PlyData, PlyElement, PlyProperty, PlyListProperty
from os import listdir, path

def convert_ascii_to_bin(file_path):
    """
    Convert ASCII PLY files to binary PLY files.

    Args:
        file_path (str): The path to the directory containing the ASCII PLY files.

    Returns:
        None
    """
    to_ascii = False

    # Check if the file path exists
    files = [f for f in listdir(file_path) if f[-4:] == '.ply']

    # Iterate through each file in the directory
    for each_file in files:

        # Load the ASCII PLY file
        print('\n Loading.... ', path.join(file_path, each_file))
        data = PlyData.read(path.join(file_path, each_file))
        print('\n Loaded..... ', path.join(file_path, each_file))

        # Define the properties of the binary PLY file
        data.elements[0].data.dtype.names = ['x', 'y', 'z', 'reflectance', 'class']
        data.elements[0].properties = (PlyProperty('x', 'float'), PlyProperty('y', 'float'),
                                       PlyProperty('z', 'float'), PlyProperty('reflectance', 'int'),
                                       PlyProperty('class', 'int'))
        
        # Write the binary PLY file
        data1 = PlyData([data.elements[0]], text=to_ascii)
        data1.write(path.join(file_path, 'bin_' + each_file))
        print('\n completed.. ', each_file)

        # Load the binary PLY file
        data2 = PlyData.read(path.join(file_path, 'bin_'+each_file))

        # Print the first element of the binary PLY file
        print(data.elements[0])
        print('\n')
