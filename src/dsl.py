# Isaac Joffe, 2026


import numpy as np
import scipy
import skimage
import copy


# Define the OBJECT data type
###############################################################################
class MyObject():
    def __init__(self, grid):
        self.set_grid(grid)
        return

    def __copy__(self):
        return MyObject(self.get_grid())

    def get_grid(self):
        return self.__grid.copy()

    def set_grid(self, grid):
        assert isinstance(grid, np.ndarray), f"Attempted to create an object out of {grid}, which is not a NumPy array"
        assert (grid.dtype == int), f"Attempted to create an object out of {grid}, which is not a NumPy array of integers"
        self.__grid = grid
        return


class Hypothesis():
    @staticmethod
    def get_parameter_type():
        return []

    @staticmethod
    def get_return_type():
        return Hypothesis


class GridHypothesis(Hypothesis):
    pass

class ColourHypothesis(Hypothesis):
    pass

class TwoConnectedHypothesis(Hypothesis):
    pass

class OneConnectedHypothesis(Hypothesis):
    pass

class PartitionHypothesis(Hypothesis):
    pass


ENCODING_CACHE = {}
#
def EncodeObjects(input_grid, hypothesis, background=0):
    input_grid = np.array(input_grid).copy()

    # Cache: Avoid re-parsing the grid into objects when this has already been done
    if (input_grid.tobytes() in ENCODING_CACHE) and (hypothesis in ENCODING_CACHE[input_grid.tobytes()]):
        return ENCODING_CACHE[input_grid.tobytes()][hypothesis]

    # Otherwise, parse as normal according to provided hypothesis
    objects = []

    if (hypothesis == "GridHypothesis"):
        input_grid[input_grid == background] = BACKGROUND
        if (np.any(input_grid != BACKGROUND)):
            objects.append(MyObject(input_grid))

    elif (hypothesis == "TwoConnectedHypothesis"):
        parsed_input_grid, n_objects = skimage.measure.label(
            input_grid,
            connectivity=2,
            background=background,
            return_num=True,
        )
        objects.extend(MyObject(input_grid * (parsed_input_grid == i + 1) + BACKGROUND * (parsed_input_grid != i + 1)) for i in range(n_objects))

    elif (hypothesis == "OneConnectedHypothesis"):
        parsed_input_grid, n_objects = skimage.measure.label(
            input_grid,
            connectivity=1,
            background=background,
            return_num=True,
        )
        objects.extend(MyObject(input_grid * (parsed_input_grid == i + 1) + BACKGROUND * (parsed_input_grid != i + 1)) for i in range(n_objects))

    elif (hypothesis == "ColourHypothesis"):
        objects.extend(MyObject(input_grid * (input_grid == c) + BACKGROUND * (input_grid != c)) for c in np.unique(input_grid[input_grid != background]))

    elif (hypothesis == "PartitionHypothesis"):
        # Try: left/right, no line
        if (input_grid.shape[1] % 2 == 0) and (all([(len(np.unique(partition[partition != background])) == 1) for partition in np.hsplit(input_grid, 2)])):
            partitions = np.hsplit(input_grid, 2)
        # Try: left/right, line
        elif (input_grid.shape[1] % 2 == 1) and (len(np.unique(input_grid[:, input_grid.shape[1] // 2])) == 1) and (all([(len(np.unique(partition[partition != background])) == 1) for partition in np.hsplit(np.delete(input_grid, input_grid.shape[1] // 2, axis=1), 2)])):
            partitions = np.hsplit(np.delete(input_grid, input_grid.shape[1] // 2, axis=1), 2)
        # Try: top/bottom, no line
        elif (input_grid.shape[0] % 2 == 0) and (all([(len(np.unique(partition[partition != background])) == 1) for partition in np.vsplit(input_grid, 2)])):
            partitions = np.vsplit(input_grid, 2)
        # Try: top/bottom, line
        elif (input_grid.shape[0] % 2 == 1) and (len(np.unique(input_grid[input_grid.shape[0] // 2, :])) == 1) and (all([(len(np.unique(partition[partition != background])) == 1) for partition in np.vsplit(np.delete(input_grid, input_grid.shape[0] // 2, axis=0), 2)])):
            partitions = np.vsplit(np.delete(input_grid, input_grid.shape[0] // 2, axis=0), 2)
        else:
            assert False, f"The input grid could not be partitioned"
        objects.extend(MyObject(partition - (background + 1) * (partition == background)) for partition in partitions)

    else:
        assert False, f"{hypothesis} is not a valid object hypothesis"

    # If there are no objects, force an empty object
    if (not objects):
        objects = [MyObject(input_grid)]

    # Cache: Update for future re-use
    if (input_grid.tobytes() not in ENCODING_CACHE):
        ENCODING_CACHE[input_grid.tobytes()] = {}
    ENCODING_CACHE[input_grid.tobytes()][hypothesis] = objects

    return objects


def DecodeObjects(output_objects, background=0):
    output_grid = np.zeros_like(output_objects[0].get_grid()) + background
    assert ((output_grid.shape[0] <= 30) and (output_grid.shape[1] <= 30)), "Output grid can be at most 30x30"
    for output_object in output_objects:
        grid = output_object.get_grid()
        assert (grid.shape == output_grid.shape), "All output objects must have the same size for decoding"
        overlap_indices = np.logical_and(grid != BACKGROUND, output_grid != background)
        assert np.all(grid[overlap_indices] == output_grid[overlap_indices]), "Output objects cannot overlap with conflicting colours"
        output_grid[grid != BACKGROUND] = grid[grid != BACKGROUND]
    return output_grid
###############################################################################


# Define the COLOUR data type
###############################################################################
N_COLOURS = 10

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CYAN = 8
PURPLE = 9

COLOURS = {
    0: "BLACK",
    1: "BLUE",
    2: "RED",
    3: "GREEN",
    4: "YELLOW",
    5: "GREY",
    6: "ORANGE",
    7: "PINK",
    8: "CYAN",
    9: "PURPLE",
}

BACKGROUND = -1

class Colour():
    def __init__(self, value):
        self.set_value(value)
        return

    def __call__(self):
        return self.__value

    def __copy__(self):
        return Colour(self())

    def set_value(self, value):
        assert ((isinstance(value, int) or (value.dtype == int)) and (value in COLOURS.keys())), f"Colour {value} is not valid"
        self.__value = value
        return
###############################################################################


# Define the NUMBER data type
###############################################################################
class Number():
    def __init__(self, value):
        self.set_value(value)
        return

    def __call__(self):
        return self.__value

    def __copy__(self):
        return Number(self())

    def set_value(self, value):
        assert (isinstance(value, int)), f"Number {value} is not valid"
        self.__value = value
        return
###############################################################################


# Define the BOOLEAN data type
###############################################################################
class Boolean():
    def __init__(self, value):
        self.set_value(value)
        return

    def __call__(self):
        return self.__value

    def __copy__(self):
        return Boolean(self())

    def set_value(self, value):
        assert (isinstance(value, bool)), f"Boolean {value} is not valid"
        self.__value = value
        return
###############################################################################


# Define operations that return the LIST[OBJECT] data type
###############################################################################
class ArgMax():
    def __init__(self, input_objects: list[MyObject], input_numbers: list[Number]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_numbers, list) and all([isinstance(input_number, Number) for input_number in input_numbers])), "Second input must be a list of numbers"
        assert (len(input_objects) == len(input_numbers)), f"Number of objects and numbers must match, but received {len(input_objects)} objects and {len(input_numbers)} numbers"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_numbers = [copy.copy(input_number)() for input_number in input_numbers]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        max_number = np.max(self.__input_numbers)
        for input_object, input_number in zip(self.__input_objects, self.__input_numbers):
            if (input_number == max_number):
                output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Number]

    @staticmethod
    def get_return_type():
        return MyObject


class ArgMin():
    def __init__(self, input_objects: list[MyObject], input_numbers: list[Number]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_numbers, list) and all([isinstance(input_number, Number) for input_number in input_numbers])), "Second input must be a list of numbers"
        assert (len(input_objects) == len(input_numbers)), f"Number of objects and numbers must match, but received {len(input_objects)} objects and {len(input_numbers)} numbers"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_numbers = [copy.copy(input_number)() for input_number in input_numbers]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        min_number = np.min(self.__input_numbers)
        for input_object, input_number in zip(self.__input_objects, self.__input_numbers):
            if (input_number == min_number):
                output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Number]

    @staticmethod
    def get_return_type():
        return MyObject


class Crop():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = crop_grid(input_grid)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Encircle():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    # TODO: Needs more optimization
    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            indices = np.array(np.where(output_grid != BACKGROUND)).T
            for index in indices:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        if (index[0] + i >= 0) and (index[0] + i < output_grid.shape[0]) and (index[1] + j >= 0) and (index[1] + j < output_grid.shape[1]) and (output_grid[index[0] + i, index[1] + j] == BACKGROUND):
                            output_grid[index[0] + i, index[1] + j] = input_colour
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject


class FillHoles():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            if (np.sum(output_grid != BACKGROUND) >= 4):    # Heuristic: Object must have at least 4 pixels to have a hole
                output_grid[scipy.ndimage.binary_fill_holes(output_grid >= 0) ^ (output_grid >= 0)] = input_colour
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject


class FillLines():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            if (np.sum(output_grid != BACKGROUND) > 1):    # Heuristic: Object must have multiple pixels to have a line to fill
                indices = []
                for i in range(output_grid.shape[0]):
                    mask = (output_grid[i] != BACKGROUND)
                    if (np.sum(mask) >= 2):
                        start = np.argmax(mask) + 1
                        end = output_grid.shape[1] - 1 - np.argmax(mask[::-1])
                        for k in range(start, end):
                            indices.append((i, k))
                for j in range(output_grid.shape[1]):
                    mask = (output_grid[:, j] != BACKGROUND)
                    if (np.sum(mask) >= 2):
                        start = np.argmax(mask) + 1
                        end = output_grid.shape[0] - 1 - np.argmax(mask[::-1])
                        for k in range(start, end):
                            indices.append((k, j))
                for index in indices:
                    if (output_grid[index] == BACKGROUND):
                        output_grid[index] = input_colour
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject


class Filter():
    def __init__(self, input_objects: list[MyObject], input_booleans: list[Boolean]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_booleans, list) and all([isinstance(input_boolean, Boolean) for input_boolean in input_booleans])), "Second input must be a list of booleans"
        assert (len(input_objects) == len(input_booleans)), f"Number of objects and booleans must match, but received {len(input_objects)} objects and {len(input_booleans)} booleans"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_booleans = [copy.copy(input_boolean)() for input_boolean in input_booleans]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_boolean in zip(self.__input_objects, self.__input_booleans):
            if (input_boolean):
                output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Boolean]

    @staticmethod
    def get_return_type():
        return MyObject


class HFlip():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = np.flip(input_grid, axis=1)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Hollow():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            output_grid = input_object.get_grid()
            if (np.sum(output_grid != BACKGROUND) >= 5):    # Heuristic: Object must have at least 5 pixels to be hollow-able
                output_grid -= scipy.ndimage.binary_erosion(output_grid >= 0) * (output_grid + 1)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class HStack():
    def __init__(self, input_objects_left: list[MyObject], input_objects_right: list[MyObject]):
        assert (isinstance(input_objects_left, list) and (len(input_objects_left) == 1) and all([isinstance(input_object, MyObject) for input_object in input_objects_left])), "First input must be a list of one object"
        assert (isinstance(input_objects_right, list) and (len(input_objects_right) == 1)  and all([isinstance(input_object, MyObject) for input_object in input_objects_right])), "Second input must be a list of one object"
        self.__input_objects_left = [copy.copy(input_object) for input_object in input_objects_left]
        self.__input_objects_right = [copy.copy(input_object) for input_object in input_objects_right]
        return

    def __call__(self) -> list[MyObject]:
        left_grid = self.__input_objects_left[0].get_grid()
        right_grid = self.__input_objects_right[0].get_grid()
        output_grid = np.hstack((left_grid, right_grid))
        output_objects = [MyObject(output_grid)]
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Identity():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class PixelwiseAnd():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        assert (np.all((input_object.shape == input_objects[0].shape) for input_object in input_objects)), "All objects must have the same shape"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_grid = self.__input_objects[0].get_grid()
        for input_object in self.__input_objects[1:]:
            input_grid = input_object.get_grid()
            output_grid[input_grid == BACKGROUND] = BACKGROUND
        output_objects = [MyObject(output_grid)]
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class PixelwiseNot():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            output_grid = input_object.get_grid()
            output_colour = Colour(get_main_colour(output_grid))()
            background = (output_grid == BACKGROUND)
            foreground = (output_grid != BACKGROUND)
            output_grid[background] = output_colour
            output_grid[foreground] = BACKGROUND
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class PixelwiseOr():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        assert (np.all((input_object.shape == input_objects[0].shape) for input_object in input_objects)), "All objects must have the same shape"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_grid = self.__input_objects[0].get_grid()
        for input_object in self.__input_objects[1:]:
            input_grid = input_object.get_grid()
            output_grid[input_grid != BACKGROUND] = input_grid[input_grid != BACKGROUND]
        output_objects = [MyObject(output_grid)]
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class PixelwiseXor():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        assert (np.all((input_object.shape == input_objects[0].shape) for input_object in input_objects)), "All objects must have the same shape"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    # TODO: Does this work for more than 2 objects? Limit to 2?
    def __call__(self) -> list[MyObject]:
        if (len(self.__input_objects) != 2):    # Heuristic: Only two objects can be XORed
            assert False, "PixelwiseXor can only accept 2 objects"
        output_grid = self.__input_objects[0].get_grid()
        for input_object in self.__input_objects[1:]:
            input_grid = input_object.get_grid()
            same = np.logical_or(np.logical_and((output_grid == BACKGROUND), (input_grid == BACKGROUND)), np.logical_and((output_grid != BACKGROUND), (input_grid != BACKGROUND)))
            different = np.logical_and((output_grid == BACKGROUND), (input_grid != BACKGROUND))
            output_grid[same] = BACKGROUND
            output_grid[different] = input_grid[different]
        output_objects = [MyObject(output_grid)]
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Recolour():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            output_grid[output_grid != BACKGROUND] = input_colour
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject


class Rectangify():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            if (np.sum(output_grid != BACKGROUND) > 1):    # Heuristic: Object must have multiple pixels to be expandable into a rectangle
                top_row, bottom_row, left_column, right_column = get_bounding_coordinates(output_grid)
                object_patch = output_grid[top_row:bottom_row, left_column:right_column]
                object_patch[object_patch == BACKGROUND] = input_colour
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject


class Rot90():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = np.rot90(input_grid, k=3)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Rot180():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = np.rot90(input_grid, k=2)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Rot270():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = np.rot90(input_grid, k=1)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Scale():
    def __init__(self, input_objects: list[MyObject], input_numbers: list[Number]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_numbers, list) and all([isinstance(input_number, Number) for input_number in input_numbers])), "Second input must be a list of numbers"
        assert ((len(input_objects) == len(input_numbers)) or (len(input_numbers) == 1)), f"Number of objects and numbers must match or there must be only one number, but received {len(input_objects)} objects and {len(input_numbers)} numbers"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_numbers = [copy.copy(input_number)() for input_number in input_numbers]
        if (len(self.__input_numbers) == 1):
            self.__input_numbers *= len(self.__input_numbers)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_number in zip(self.__input_objects, self.__input_numbers):
            input_grid = input_object.get_grid()
            output_grid = input_grid.repeat(input_number, axis=0).repeat(input_number, axis=1)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Number]

    @staticmethod
    def get_return_type():
        return MyObject


class Transpose():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = input_grid.T
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class VFlip():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_grid = np.flip(input_grid, axis=0)
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class VStack():
    def __init__(self, input_objects_top: list[MyObject], input_objects_bottom: list[MyObject]):
        assert (isinstance(input_objects_top, list) and (len(input_objects_top) == 1) and all([isinstance(input_object, MyObject) for input_object in input_objects_top])), "First input must be a list of one object"
        assert (isinstance(input_objects_bottom, list) and (len(input_objects_bottom) == 1)  and all([isinstance(input_object, MyObject) for input_object in input_objects_bottom])), "Second input must be a list of one object"
        self.__input_objects_top = [copy.copy(input_object) for input_object in input_objects_top]
        self.__input_objects_bottom = [copy.copy(input_object) for input_object in input_objects_bottom]
        return

    def __call__(self) -> list[MyObject]:
        top_grid = self.__input_objects_top[0].get_grid()
        bottom_grid = self.__input_objects_bottom[0].get_grid()
        output_grid = np.vstack((top_grid, bottom_grid))
        output_objects = [MyObject(output_grid)]
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, MyObject]

    @staticmethod
    def get_return_type():
        return MyObject


class Xify():
    def __init__(self, input_objects: list[MyObject], input_colours: list[Colour]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "First input must be a list of objects"
        assert (isinstance(input_colours, list) and all([isinstance(input_colour, Colour) for input_colour in input_colours])), "Second input must be a list of colours"
        assert ((len(input_objects) == len(input_colours)) or (len(input_colours) == 1)), f"Number of objects and colours must match or there must be only one colour, but received {len(input_objects)} objects and {len(input_colours)} colours"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        self.__input_colours = [copy.copy(input_colour)() for input_colour in input_colours]
        if (len(self.__input_colours) == 1):
            self.__input_colours *= len(self.__input_objects)
        return

    def __call__(self) -> list[MyObject]:
        output_objects = []
        for input_object, input_colour in zip(self.__input_objects, self.__input_colours):
            output_grid = input_object.get_grid()
            if (np.sum(output_grid != BACKGROUND) > 1):    # Heuristic: Object must be exactly 1 pixel to be expandable into an X
                assert False, "Xify can only accept objects with 1 pixel"
            index = np.array(np.where(output_grid != BACKGROUND)).flatten()
            down_diagonal = np.array([max(0, index[0] - index[1]), max(0, index[1] - index[0])])
            while True:
                if (down_diagonal[0] >= output_grid.shape[0]) or (down_diagonal[1] >= output_grid.shape[1]):
                    break
                if np.any(index != down_diagonal):
                    output_grid[down_diagonal[0], down_diagonal[1]] = input_colour
                down_diagonal += 1
            up_diagonal = np.array([output_grid.shape[0] - 1 - max(0, output_grid.shape[0] - 1 - index[0] - index[1]), max(0, index[1] + index[0] + 1 - output_grid.shape[0])])
            while True:
                if (up_diagonal[0] < 0) or (up_diagonal[1] >= output_grid.shape[1]):
                    break
                if np.any(index != up_diagonal):
                    output_grid[up_diagonal[0], up_diagonal[1]] = input_colour
                up_diagonal[0] -= 1
                up_diagonal[1] += 1
            input_object.set_grid(output_grid)
            output_objects.append(input_object)
        return output_objects

    @staticmethod
    def get_parameter_type():
        return [MyObject, Colour]

    @staticmethod
    def get_return_type():
        return MyObject
###############################################################################


# Define operations that return the LIST[COLOUR] data type
###############################################################################
class ColourConstant():
    def __init__(self, input_colour: Colour):
        assert isinstance(input_colour, Colour), "Input must be a colour"
        self.__input_colour = input_colour
        return

    def __call__(self) -> list[Colour]:
        return [self.__input_colour]

    @staticmethod
    def get_parameter_type():
        return []

    @staticmethod
    def get_return_type():
        return Colour


class ObjectGetColour():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Colour]:
        output_colours = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_colour = Colour(get_main_colour(input_grid))
            output_colours.append(output_colour)
        return output_colours

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Colour
###############################################################################


# Define operations that return the LIST[NUMBER] data type
###############################################################################
class NumberConstant():
    def __init__(self, input_number: Number):
        assert isinstance(input_number, Number), "Input must be a number"
        self.__input_number = input_number
        return

    def __call__(self) -> list[Number]:
        return [self.__input_number]

    @staticmethod
    def get_parameter_type():
        return []

    @staticmethod
    def get_return_type():
        return Number


class ObjectGetColourCount():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Number]:
        output_numbers = []
        for input_object in self.__input_objects:
            grid = input_object.get_grid()
            output_number = Number(len(np.unique(grid[grid != BACKGROUND])))
            output_numbers.append(output_number)
        return output_numbers

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Number


class ObjectGetHeight():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Number]:
        output_numbers = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            input_grid = crop_grid(input_grid)
            output_number = Number(int(input_grid.shape[0]))
            output_numbers.append(output_number)
        return output_numbers

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Number


class ObjectGetPixelCount():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Number]:
        output_numbers = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_number = Number(int(np.sum(input_grid != BACKGROUND)))
            output_numbers.append(output_number)
        return output_numbers

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Number


class ObjectGetWidth():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Number]:
        output_numbers = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            input_grid = crop_grid(input_grid)
            output_number = Number(int(input_grid.shape[1]))
            output_numbers.append(output_number)
        return output_numbers

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Number
###############################################################################


# Define operations that return the LIST[BOOLEAN] data type
###############################################################################
class ObjectHasHoles():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Boolean]:
        output_booleans = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            output_boolean = Boolean(bool(np.any(scipy.ndimage.binary_fill_holes(input_grid != BACKGROUND) ^ (input_grid != BACKGROUND))))
            output_booleans.append(output_boolean)
        return output_booleans

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Boolean


class ObjectIsHSymmetrical():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Boolean]:
        output_booleans = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            input_grid = crop_grid(input_grid)
            output_boolean = Boolean(bool(np.all(input_grid == np.flip(input_grid, axis=1))))
            output_booleans.append(output_boolean)
        return output_booleans

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Boolean


class ObjectIsVSymmetrical():
    def __init__(self, input_objects: list[MyObject]):
        assert (isinstance(input_objects, list) and all([isinstance(input_object, MyObject) for input_object in input_objects])), "Input must be a list of objects"
        self.__input_objects = [copy.copy(input_object) for input_object in input_objects]
        return

    def __call__(self) -> list[Boolean]:
        output_booleans = []
        for input_object in self.__input_objects:
            input_grid = input_object.get_grid()
            input_grid = crop_grid(input_grid)
            output_boolean = Boolean(bool(np.all(input_grid == np.flip(input_grid, axis=0))))
            output_booleans.append(output_boolean)
        return output_booleans

    @staticmethod
    def get_parameter_type():
        return [MyObject]

    @staticmethod
    def get_return_type():
        return Boolean
###############################################################################


# HELPER FUNCTIONS
###############################################################################
def get_bounding_coordinates(grid):
    top_row = np.argmax((grid != BACKGROUND).any(axis=1))
    bottom_row = grid.shape[0] - np.argmax((grid != BACKGROUND).any(axis=1)[::-1])
    left_column = np.argmax((grid != BACKGROUND).any(axis=0))
    right_column = grid.shape[1] - np.argmax((grid != BACKGROUND).any(axis=0)[::-1])
    return top_row, bottom_row, left_column, right_column


def crop_grid(grid):
    top_row, bottom_row, left_column, right_column = get_bounding_coordinates(grid)
    cropped_grid = grid[top_row:bottom_row, left_column:right_column].copy()
    return cropped_grid


def get_main_colour(grid):
    colours, counts = np.unique(grid, return_counts=True)
    main_colour = colours[(np.argmax(counts[1:]) + 1) if (BACKGROUND in colours) else (np.argmax(counts))]
    return main_colour
###############################################################################


def main():
    return


if __name__ == "__main__":
    main()
