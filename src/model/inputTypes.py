from enum import Enum

class InputTypes(Enum):
    Normal = 1 # Combine VH0, VH1, VV0, VV1, dem through an algorithm
    VV = 2 # Use VV0, VV1, dem
    VH = 3 # Use VH0, VH1, slope
    All = 4 # Use all channels: VH0, VH1, VV0, VV1, dem, slope
    Original = 5 # Use original RGB images
    Difference = 6 # Combine VH0, VH1, VV0, VV1, dem: (VH1-VH0) and (VV1-VV0) differences