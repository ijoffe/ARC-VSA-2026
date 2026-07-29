# Isaac Joffe, 2026


from utils import *
from search import *


BASE_SEED = 0
BASE_RNG = reseed(BASE_SEED)


# Helper functions
###############################################################################
def generate_random_dimensions(max_size=25, shape=""):
    dim_1 = np.random.randint(3, max_size)
    dim_2 = int(np.random.normal(dim_1, dim_1 / (4 if (shape == "squarish") else 2)))
    dimensions = np.maximum(3, np.minimum(30, np.array([dim_1, dim_2])))
    np.random.shuffle(dimensions)
    return dimensions


def generate_random_colour():
    return list(COLOURS.values())[np.random.randint(1, N_COLOURS)]


def generate_random_number():
    return np.random.randint(10) + 1


def generate_grid_for_grid(mode=""):
    if (mode.lower() == "small"):
        dimensions = generate_random_dimensions(max_size=6)
    else:
        dimensions = generate_random_dimensions()

    input_grid = np.random.choice(
        N_COLOURS,
        dimensions,
        p=(
            np.hstack((0.5, np.random.permutation([0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15, 0.15])))
                if (mode == "background") else
            np.random.permutation([0, 0, 0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4])
        ),
    )
    return input_grid


def generate_grid_for_connectedobjects(mode=""):
    dimensions = generate_random_dimensions(shape="squarish")

    input_grid = skimage.draw.random_shapes(
        dimensions,
        shape=None,
        min_shapes=(5 if (mode == "busy") else 1),
        max_shapes=(10 if (mode == "busy") else np.random.randint(1, 10)),
        min_size=2,
        max_size=max(dimensions),
        allow_overlap=(mode == "busy"),
        channel_axis=None,
        intensity_range=(1, 9),
        rng=BASE_RNG,
    )[0]
    input_grid = input_grid - 255 * (input_grid == 255)
    return input_grid


def generate_grid_for_colour():
    dimensions = generate_random_dimensions()

    if (np.random.choice([True, False])):
        input_grid = np.random.choice(
            N_COLOURS,
            dimensions,
            p=np.concat(([0.9], np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0, 0.1]))),
        )
    else:
        input_grid = np.random.choice(
            N_COLOURS,
            dimensions,
            p=np.concat(([0.9], np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0.05, 0.05]))),
        )
    return input_grid


def generate_grid_for_partition(mode=""):
    dimensions = generate_random_dimensions(max_size=15)

    probability = np.maximum(0, np.minimum(1, np.random.normal(0.5, 0.1, size=2)))
    parts = np.random.choice(
        N_COLOURS,
        dimensions,
        p=np.concat(([1 - probability[0]], np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0, probability[0]]))),
    ), np.random.choice(
        N_COLOURS,
        dimensions,
        p=np.concat(([1 - probability[1]], np.random.permutation([0, 0, 0, 0, 0, 0, 0, 0, probability[1]]))),
    )

    if (mode == "part"):
        same = np.logical_or(np.logical_and((parts[0] == 0), (parts[1] == 0)), np.logical_and((parts[0] != 0), (parts[1] != 0)))
        different = np.logical_and((parts[0] == 0), (parts[1] != 0))
        input_grid = parts[0]
        input_grid[same] = 0
        input_grid[different] = parts[1][different]
        return input_grid

    if (np.random.choice([True, False])):
        if (np.random.choice([True, False])):
            input_grid = np.hstack(parts)
        else:
            input_grid = np.hstack((parts[0], np.full((dimensions[0], 1), np.random.randint(1, N_COLOURS)), parts[1]))
    else:
        if (np.random.choice([True, False])):
            input_grid = np.vstack(parts)
        else:
            input_grid = np.vstack((parts[0], np.full((1, dimensions[1]), np.random.randint(1, N_COLOURS)), parts[1]))

    return input_grid


def generate_grid_for_single():
    dimensions = generate_random_dimensions()

    input_grid = skimage.draw.random_shapes(
        dimensions,
        shape="circle",
        min_shapes=1,
        max_shapes=np.random.randint(1, 5),
        min_size=2,
        max_size=2,
        allow_overlap=False,
        channel_axis=None,
        intensity_range=(1, 9),
        rng=BASE_RNG,
    )[0]
    input_grid = input_grid - 255 * (input_grid == 255)
    return input_grid
###############################################################################


# Generate sample grids and programs
###############################################################################
# Generate a dataset of simple (i.e., 'one-level') programs
def generate_shallow_samples(n_per_type):
    input_grids = []
    output_grids = []
    programs = []

    # Generate n input-output grid pairs for each object-producing primitive
    primitives = [p for p in SORTED_PRIMITIVES if (p not in [
        GridHypothesis,
        ColourHypothesis,
        TwoConnectedHypothesis,
        OneConnectedHypothesis,
        PartitionHypothesis,
        ColourConstant,
        ObjectGetColour,
        ArgMax,
        ArgMin,
        Filter,
        Scale,
    ])]
    for p in tqdm(primitives):
        for _ in range(n_per_type):

            # Need to keep iterating until a valid grid-program pair found
            while True:
                try:

                    # Grid-based primitives
                    if (p in [Crop]):
                        object_hypothesis = np.random.choice([GridHypothesis, ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_connectedobjects()

                        program = (f"{object_hypothesis.__name__}; {p.__name__}(x)()", [object_hypothesis, p])

                    elif (p in [Rot90, Rot180, Rot270, HFlip, VFlip, Transpose]):
                        object_hypothesis = GridHypothesis
                        if (np.random.choice([True, False], p=[0.5, 0.5])):
                            input_grid = generate_grid_for_grid()
                        else:
                            input_grid = generate_grid_for_connectedobjects(mode="busy")

                        program = (f"{object_hypothesis.__name__}; {p.__name__}(x)()", [object_hypothesis, p])

                    elif (p in [HStack, VStack]):
                        object_hypothesis = GridHypothesis
                        if (np.random.choice([True, False], p=[0.5, 0.5])):
                            input_grid = generate_grid_for_grid()
                        else:
                            input_grid = generate_grid_for_connectedobjects(mode="busy")

                        program = (f"{object_hypothesis.__name__}; {p.__name__}(x, x)()", [object_hypothesis, p])

                    elif (p in [ObjectGetColourCount]):
                        object_hypothesis = GridHypothesis
                        input_grid = generate_grid_for_grid(mode="small")

                        program = (f"{object_hypothesis.__name__}; Scale(x, {p.__name__}(x)())()", [object_hypothesis, Scale, p])

                    elif (p in [NumberConstant]):
                        object_hypothesis = GridHypothesis
                        input_grid = generate_grid_for_grid(mode="small")

                        program = (f"{object_hypothesis.__name__}; Scale(x, NumberConstant(Number({generate_random_number()}))())()", [object_hypothesis, Scale, NumberConstant])

                    # Object-transform-based primitives
                    elif (p in [Encircle]):
                        object_hypothesis = np.random.choice([GridHypothesis, ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        if (np.random.choice([True, False], p=[0.5, 0.5])):
                            input_grid = generate_grid_for_connectedobjects()
                        else:
                            input_grid = generate_grid_for_single()

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    elif (p in [FillHoles]):
                        object_hypothesis = np.random.choice([TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_connectedobjects()
                        input_grid[scipy.ndimage.binary_erosion(input_grid > 0)] = 0

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    elif (p in [FillLines]):
                        object_hypothesis = ColourHypothesis
                        input_grid = generate_grid_for_colour()

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    elif (p in [Hollow]):
                        object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_connectedobjects()

                        program = (f"{object_hypothesis.__name__}; {p.__name__}(x)()", [object_hypothesis, p])

                    elif (p in [Recolour]):
                        object_hypothesis = np.random.choice([GridHypothesis, ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        if (np.random.choice([True, False])):
                            input_grid = generate_grid_for_connectedobjects()
                        else:
                            input_grid = generate_grid_for_grid(mode="background")

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    elif (p in [Rectangify]):
                        object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        if (np.random.choice([True, False])):
                            input_grid = generate_grid_for_connectedobjects()
                        else:
                            input_grid = generate_grid_for_colour()

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    elif (p in [Xify]):
                        object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_single()

                        if (np.random.choice([True, False])):
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ColourConstant(Colour({generate_random_colour()}))())()", [object_hypothesis, p, ColourConstant])
                        else:
                            program = (f"{object_hypothesis.__name__}; {p.__name__}(x, ObjectGetColour(x)())()", [object_hypothesis, p, ObjectGetColour])

                    # Partition-based primitives
                    elif (p in [PixelwiseAnd, PixelwiseOr, PixelwiseXor]):
                        object_hypothesis = PartitionHypothesis
                        input_grid = generate_grid_for_partition()

                        program = (f"{object_hypothesis.__name__}; {p.__name__}(x)()", [object_hypothesis, p])

                    elif (p in [PixelwiseNot]):
                        object_hypothesis = PartitionHypothesis
                        input_grid = generate_grid_for_partition(mode="part")

                        program = (f"GridHypothesis; {p.__name__}(x)()", [object_hypothesis, p])

                    # Selection-based primitives
                    elif (p in [ObjectGetPixelCount, ObjectGetHeight, ObjectGetWidth]):
                        if (np.random.choice([True, False], p=[0.9, 0.1])):
                            object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                            if (np.random.choice([True, False], p=[0.9, 0.1])):
                                input_grid = generate_grid_for_connectedobjects()
                            else:
                                input_grid = generate_grid_for_grid(mode="background")

                            subprogram = np.random.choice([ArgMin, ArgMax])
                            program = (f"{object_hypothesis.__name__}; {subprogram.__name__}(x, {p.__name__}(x)())()", [object_hypothesis, subprogram, p])

                        else:
                            object_hypothesis = GridHypothesis
                            input_grid = generate_grid_for_grid(mode="small")

                            program = (f"{object_hypothesis.__name__}; Scale(x, {p.__name__}(x)())()", [object_hypothesis, Scale, p])

                    elif (p in [ObjectIsHSymmetrical, ObjectIsVSymmetrical]):
                        object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_connectedobjects()

                        subprogram = np.random.choice([Filter])
                        program = (f"{object_hypothesis.__name__}; {subprogram.__name__}(x, {p.__name__}(x)())()", [object_hypothesis, subprogram, p])

                    elif (p in [ObjectHasHoles]):
                        object_hypothesis = np.random.choice([ColourHypothesis, TwoConnectedHypothesis, OneConnectedHypothesis])
                        input_grid = generate_grid_for_connectedobjects()
                        input_grid[scipy.ndimage.binary_erosion(input_grid > 0)] = 0

                        subprogram = np.random.choice([Filter])
                        program = (f"{object_hypothesis.__name__}; {subprogram.__name__}(x, {p.__name__}(x)())()", [object_hypothesis, subprogram, p])

                    else:
                        assert False, f"Invalid primitive {p}"

                    program = MyProgram(*program)
                    output_grid = safe_execute_program(input_grid, program)

                    assert (np.any(input_grid.shape != output_grid.shape) or np.any(input_grid != output_grid)), f"Error in program \"{program}\": Input and output grids are identical; program does nothing"
                    assert ((input_grid.shape[0] <= 30) and (input_grid.shape[1] <= 30)), f"Error in program \"{program}\": Input grid is too large"
                    assert (np.sum(input_grid) != 0), f"Error in program \"{program}\": Input grid is empty"
                    assert ((output_grid.shape[0] <= 30) and (output_grid.shape[1] <= 30)), f"Error in program \"{program}\": Output grid is too large"
                    assert (np.sum(output_grid) != 0), f"Error in program \"{program}\": Output grid is empty"
                    break

                except Exception as e:
                    if (DEBUG):
                        print(f"Exception: {e}")
                    continue

            if (DEBUG):
                print_grid(input_grid)
                print(program)
                print_grid(output_grid)
                input()

            input_grids.append(input_grid)
            output_grids.append(output_grid)
            programs.append(program)

    assert (len(input_grids) == len(output_grids))
    return input_grids, output_grids, programs
###############################################################################


#
###############################################################################
def encode_grids(input_grids, output_grids, grid_encoding_function):
    input_data = []
    for (input_grid, output_grid) in tqdm(zip(input_grids, output_grids)):
        input_data.append(np.concat((grid_encoding_function(input_grid), grid_encoding_function(output_grid)), axis=1).flatten())
    return np.array(input_data)


def encode_primitives(programs):
    output_data = []
    for program in tqdm(programs):
        output_data.append(np.array([int(p in program.get_primitives()) for p in SORTED_PRIMITIVES]))
    return np.array(output_data)


def encode_constants(programs, constant_encoding_function):
    output_data = []
    for program in tqdm(programs):
        output_data.append(constant_encoding_function(program).flatten())
    return np.array(output_data)


def generate_dataset(n):
    print(f"Generating a dataset with >={n} programs per primitive...")
    input_grids, output_grids, programs = generate_shallow_samples(n)

    print(f"Number of occurrences of each primitive: {encode_primitives(programs).sum(axis=0)}")
    assert (np.all(encode_primitives(programs).sum(axis=0) >= n)), f"Dataset must have at least {n} examples of each primitive"

    dataset = []
    for (input_grid, output_grid, program) in zip(input_grids, output_grids, programs):
        dataset.append({
            "input_grid": input_grid.tolist(),
            "output_grid": output_grid.tolist(),
            "program": program.get_program(),
            "primitives": [p.__name__ for p in program.get_primitives()],
        })

    dataset_path = f"{DATA_PATH}New_{n}N_{BASE_SEED}S.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f)
    print(f"Dataset saved at \"{dataset_path}\"")
    exit()
    return
###############################################################################


def main():
    N_SAMPLES_PER_PRIMITIVE = [5, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    for n in tqdm(N_SAMPLES_PER_PRIMITIVE):
        generate_dataset(n)
    return


if __name__ == "__main__":
    main()
