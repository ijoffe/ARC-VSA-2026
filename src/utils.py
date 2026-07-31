# Isaac Joffe, 2026


# GENERAL SETUP
###############################################################################
# Import required system-level libraries
import itertools
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import random
import sklearn
import sys
import torch
from tqdm import tqdm

# Use my custom DSL
from dsl import *

# Use up-to-date sspspace library
sys.path.append("..")
from inc.sspspace import sspspace

# Setup directories
FIGURES_PATH = "../figures/"
DATA_PATH = "../data/"
MODELS_PATH = "../models/"
TASKS_PATH = "../tasks/"

# Use torch with CUDA if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")
if (device == "cuda"):
    print(f"\tTorch GPU count: {torch.cuda.device_count()}")

# Ensure reproducibility
def reseed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return rng

# Print detailed logs when set
DEBUG = False
###############################################################################


# TASK SETUP
###############################################################################
# Recursively find all .json task files in all subdirectories of a given directory
def get_task_names(directory):
    task_names = []
    for name in os.listdir(directory):
        path = f"{directory}/{name}"
        if ((os.path.isfile(path)) and (path.endswith(".json"))):
            task_names.append(path)
        elif (os.path.isdir(path)):
            task_names.extend(get_task_names(path))
        else:
            print(f"WARNING: Invalid path {path}")
    return task_names


# Store the task data for each task to attempt
def get_tasks(task_names):
    tasks = {}
    for task_name in task_names:
        with open(task_name, "r") as f:
            tasks[task_name] = json.load(f)
    return tasks


# Store the demonstration grids for a task as NumPy arrays
def get_demonstrations(task):
    train_ins = [np.array(demonstration["input"]).copy() for demonstration in task["train"]]
    train_outs = [np.array(demonstration["output"]).copy() for demonstration in task["train"]]
    return train_ins, train_outs


# Store the query grids for a task as NumPy arrays
def get_queries(task):
    test_ins = [np.array(query["input"]).copy() for query in task["test"]]
    test_outs = [np.array(query["output"]).copy() for query in task["test"]]
    return test_ins, test_outs


# A selection of solvable tasks
TASK_NAMES = [
    # Simple grid-operation-based tasks
    "1cf80156",
    "28bf18c6",
    "3c9b0459",
    "4c4377d9",
    "6150a2bd",
    "67a3c6ac",
    "68b16354",
    "6d0aefbc",
    "6fa7a44f",
    "7468f01a",
    "74dd1130",
    "8be77c9e",
    "9172f3a0",
    "9dfd6313",
    "a416b8f3",
    "ac0a08a4",
    "b91ae062",
    "c59eb873",
    "c9e6f938",
    "ed36ccf7",
    "f25fbde4",

    # Simple object-operation-based tasks
    "00d62c1b",
    "22168020",
    "22eb0ac0",
    "253bf280",
    "3aa6fb7a",
    "4258a5f9",
    "4347f46a",
    "50cb2852",
    "56ff96f3",
    "60b61512",
    "623ea044",
    "6d75e8bb",
    "6f8cd79b",
    "a5313dff",
    "bb43febb",
    "ce22a75a",
    "dbc1a6ce",
    "ded97339",

    # Simple object-combination-based tasks
    "0520fde7",
    "3428a4f5",
    "99b1bc43",
    "ce4f8723",
    "dae9d2b5",
    "e98196ab",

    # Simple object-selection-based tasks
    "0b148d64",
    "1f85a75f",
    "23b5c85d",
    "25d8a9c8",
    "31aa019c",
    "72ca375d",
    "a87f7484",
    "be94b721",
]
TASK_NAMES = [f"{TASKS_PATH}arcagi1/training/{t}.json" for t in TASK_NAMES]

# TASK_NAMES = [
#     "195ba7dc",
#     "31d5ba1a",
#     "34b99a2b",
#     "506d28a5",
#     "5d2a5c43",
#     "60c09cac",
#     "84db8fc4",
#     "aa18de87",
#     "bf89d739",
#     "d19f7514",
#     "d4b1c2b1",
#     "d56f2372",
#     "e133d23d",
#     "e7639916",
# ]
# TASK_NAMES = [f"{TASKS_PATH}arcagi1/evaluation/{t}.json" for t in TASK_NAMES]

# TASK_NAMES = get_task_names(TASKS_PATH + "sortofarc")
# TASK_NAMES = get_task_names(TASKS_PATH + "1darc")
# TASK_NAMES = get_task_names(TASKS_PATH + "kidsarc")
# TASK_NAMES = get_task_names(TASKS_PATH + "conceptarc")
# TASK_NAMES = get_task_names(TASKS_PATH + "miniarc")

TASKS = get_tasks(TASK_NAMES)
###############################################################################


# GRID SETUP
################################################################################
GRID_REPRESENTATION_DIMENSIONS = 1024
PROGRAM_REPRESENTATION_DIMENSIONS = 4096

GRID_POSITION_SSP_LENGTHSCALE = 0.4
GRID_NUMBER_SSP_LENGTHSCALE = 1
PROGRAM_NUMBER_SSP_LENGTHSCALE = 1

GRID_POSITION_SSP_SPACES = {}
GRID_NUMBER_SSP_SPACES = {}
GRID_COLOUR_SP_SPACES = {}
GRID_SLOT_SP_SPACES = {}
GRID_LOCATION_SPACES = {}

PROGRAM_NUMBER_SSP_SPACES = {}
PROGRAM_COLOUR_SP_SPACES = {}
PROGRAM_SLOT_SP_SPACES = {}

SEEDS = [18, 19, 20, 21, 22]
for seed in SEEDS:
    rng = reseed(seed)

    GRID_POSITION_SSP_SPACES[seed] = sspspace.RandomSSPSpace(
        domain_dim=2,
        ssp_dim=GRID_REPRESENTATION_DIMENSIONS,
        length_scale=GRID_POSITION_SSP_LENGTHSCALE,
        kernel="gaussian",
        rng=rng,
    )

    GRID_NUMBER_SSP_SPACES[seed] = sspspace.RandomSSPSpace(
        domain_dim=1,
        ssp_dim=GRID_REPRESENTATION_DIMENSIONS,
        length_scale=GRID_NUMBER_SSP_LENGTHSCALE,
        kernel="gaussian",
        rng=rng,
    )

    GRID_COLOUR_SP_SPACES[seed] = sspspace.DiscreteSPSpace(
        keys=list(COLOURS.values()) + ["FOREGROUND", "BACKGROUND"],
        ssp_dim=GRID_REPRESENTATION_DIMENSIONS,
    )

    GRID_SLOT_SP_SPACES[seed] = sspspace.DiscreteSPSpace(
        keys=["SLOT_1", "SLOT_2", "SLOT_3"],
        ssp_dim=GRID_REPRESENTATION_DIMENSIONS,
    )

    GRID_LOCATION_SPACES[seed] = {}
    for i in range(-29, 30):
        for j in range(-29, 30):
            GRID_LOCATION_SPACES[seed][(i / 2, j / 2)] = GRID_POSITION_SSP_SPACES[seed].encode([i / 2, j / 2])

    PROGRAM_NUMBER_SSP_SPACES[seed] = sspspace.RandomSSPSpace(
        domain_dim=1,
        ssp_dim=PROGRAM_REPRESENTATION_DIMENSIONS,
        length_scale=PROGRAM_NUMBER_SSP_LENGTHSCALE,
        kernel="gaussian",
        rng=rng,
    )

    PROGRAM_COLOUR_SP_SPACES[seed] = sspspace.DiscreteSPSpace(
        keys=list(COLOURS.values()),
        ssp_dim=PROGRAM_REPRESENTATION_DIMENSIONS,
    )

    PROGRAM_SLOT_SP_SPACES[seed] = sspspace.DiscreteSPSpace(
        keys=["COLOUR", "NUMBER"],
        ssp_dim=PROGRAM_REPRESENTATION_DIMENSIONS,
    )


# Encode grid into VSA representation based on foreground-background
def encode_grid_1(seed, grid):
    representation = np.zeros((1, GRID_REPRESENTATION_DIMENSIONS))
    if (grid.size == 0):
        return representation
    
    if (np.any(grid == 0)):
        representation += sspspace.bind(
            GRID_COLOUR_SP_SPACES[seed].encode(np.array(["BACKGROUND"])),
            np.sum(
                np.array([
                    GRID_LOCATION_SPACES[seed][tuple(i)]
                    for i in np.array([
                        np.where(grid == 0)[1] + 0.5 - grid.shape[1] / 2,
                        grid.shape[0] / 2 - 0.5 - np.where(grid == 0)[0],
                    ]).T
                ]),
                axis=0,
            ),
        )
    if (np.any(grid > 0)):
        representation += sspspace.bind(
            GRID_COLOUR_SP_SPACES[seed].encode(np.array(["FOREGROUND"])),
            np.sum(
                np.array([
                    GRID_LOCATION_SPACES[seed][tuple(i)]
                    for i in np.array([
                        np.where(grid > 0)[1] + 0.5 - grid.shape[1] / 2,
                        grid.shape[0] / 2 - 0.5 - np.where(grid > 0)[0],
                    ]).T
                ]),
                axis=0,
            ),
        )
    representation = sspspace.bind(
        representation,
        GRID_SLOT_SP_SPACES[seed].encode(np.array(["SLOT_1"])),
    )

    representation = normalize(representation)
    assert (representation.shape == (1, GRID_REPRESENTATION_DIMENSIONS))
    return representation


# Encode grid into VSA representation based on the colour pattern in the foreground
def encode_grid_2(seed, grid):
    representation = np.zeros((1, GRID_REPRESENTATION_DIMENSIONS))
    if (grid.size == 0):
        return representation

    colours = np.unique((grid if (np.all(grid == 0)) else grid[grid != 0]))
    for colour in colours:
        representation += sspspace.bind(
            GRID_COLOUR_SP_SPACES[seed].encode(np.array([COLOURS[colour]])),
            np.sum(
                np.array([
                    GRID_LOCATION_SPACES[seed][tuple(i)]
                    for i in np.array([
                        np.where(grid == colour)[1] + 0.5 - grid.shape[1] / 2,
                        grid.shape[0] / 2 - 0.5 - np.where(grid == colour)[0],
                    ]).T
                ]),
                axis=0,
            ),
        )
    representation = sspspace.bind(
        representation,
        GRID_SLOT_SP_SPACES[seed].encode(np.array(["SLOT_2"])),
    )

    representation = normalize(representation)
    assert (representation.shape == (1, GRID_REPRESENTATION_DIMENSIONS))
    return representation


# Encode grid into VSA representation based on the colour counts in the foreground
def encode_grid_3(seed, grid):
    representation = np.zeros((1, GRID_REPRESENTATION_DIMENSIONS))
    if (grid.size == 0):
        return representation

    colours, counts = np.unique((grid if (np.all(grid == 0)) else grid[grid != 0]), return_counts=True)
    for colour, count in zip(colours, counts):
        representation += sspspace.bind(
            GRID_COLOUR_SP_SPACES[seed].encode(np.array([COLOURS[colour]])),
            GRID_NUMBER_SSP_SPACES[seed].encode(count),
        )
    representation = sspspace.bind(
        representation,
        GRID_SLOT_SP_SPACES[seed].encode(np.array(["SLOT_3"])),
    )

    representation = normalize(representation)
    assert (representation.shape == (1, GRID_REPRESENTATION_DIMENSIONS))
    return representation


# Encode grid into a VSA representation combining different inductive biases
def encode_grid_all(seed, grid):
    return normalize(encode_grid_1(seed, grid) + encode_grid_2(seed, grid) + encode_grid_3(seed, grid))


# Encode grid into a VSA representation naively
def encode_grid_naive(seed, grid):
    representation = np.zeros((1, GRID_REPRESENTATION_DIMENSIONS))
    if (grid.size > 0):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                representation += sspspace.bind(
                    GRID_COLOUR_SP_SPACES[seed].encode(np.array([COLOURS[grid[i, j]]])),
                    GRID_POSITION_SSP_SPACES[seed].encode([j + 0.5 - grid.shape[1] / 2, grid.shape[0] / 2 - 0.5 - i]),
                )
    assert (representation.shape == (1, GRID_REPRESENTATION_DIMENSIONS))
    return representation


# Decode VSA representation into a grid naively
def decode_grid_naive(seed, representation, shape):
    assert (representation.shape == (1, GRID_REPRESENTATION_DIMENSIONS))
    grid = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid[i][j] = np.argmax(sspspace.bind(
                representation,
                sspspace.invert(GRID_POSITION_SSP_SPACES[seed].encode([j + 0.5 - shape[1] / 2, shape[0] / 2 - 0.5 - i])),
            ) @ GRID_COLOUR_SP_SPACES[seed].map.T)
    return grid


# Visualize grid as text
destination = "terminal"
TEXT_COLOURS = {
    BLACK: "\N{Black Large Square}",
    BLUE: "\N{Large Blue Square}",
    RED: "\N{Large Red Square}",
    GREEN: "\N{Large Green Square}",
    YELLOW: "\N{Large Yellow Square}",
    GREY: "\U0001FA76" + (" " if (destination == "terminal") else ""),
    PINK: "\U0001FA77" + (" " if (destination == "terminal") else ""),
    ORANGE: "\N{Large Orange Square}",
    CYAN: "\U0001FA75" + (" " if (destination == "terminal") else ""),
    PURPLE: "\N{Large Purple Square}",
}
def print_grid(grid):
    print("+" + "--" * grid.shape[1] + "+")
    for i in range(grid.shape[0]):
        print("|", end="")
        for j in range(grid.shape[1]):
            print(TEXT_COLOURS[grid[i, j]], end="")
        print("|")
    print("+" + "--" * grid.shape[1] + "+")
    return
###############################################################################


# PROGRAM SETUP
###############################################################################
class MyProgram():
    def __init__(self, program, primitives, score=0, representation=None):
        self.__program = program
        self.__primitives = primitives
        self.set_score(score)
        self.__representation = representation
        return

    def __str__(self):
        return self.get_program()

    def get_program(self):
        return self.__program

    def get_program_hypothesis(self):
        return self.__program.split("; ")[0]

    def get_program_body(self):
        return self.__program.split("; ")[1]

    def get_primitives(self):
        return self.__primitives

    def get_score(self):
        return self.__score

    def set_score(self, score):
        assert (isinstance(score, float) or isinstance(score, int)), f"Program score \"{score}\" must be numeric"
        self.__score = score
        return

    def compute_representation(self, embeddings, normalized=False):
        representation = np.sum([embeddings[SORTED_PRIMITIVES.index(p)] for p in self.get_primitives()], axis=0)
        if (normalized):
            return normalize(representation)
        return representation

    def copy_with_score(self, score=None):
        return MyProgram(
            self.get_program(),
            self.get_primitives(),
            score if (score is not None) else self.get_score(),
        )


# Helper function to define an empty program
def get_null_program():
    return MyProgram("", [])


# Encodes constants used in programs
def encode_constant(seed, program):
    constants_representation = np.zeros((1, PROGRAM_REPRESENTATION_DIMENSIONS))

    colour_constants = program.get_program_body().split("ColourConstant(Colour(")
    for i in range(program.get_program_body().count("ColourConstant")):
        colour_constant = colour_constants[i + 1].split(")")[0]    # (i + 1)th substring contains "x))())..."
        constants_representation += sspspace.bind(
            PROGRAM_COLOUR_SP_SPACES[seed].encode(np.array([colour_constant])),
            PROGRAM_SLOT_SP_SPACES[seed].encode(np.array(["COLOUR"])),
        )

    number_constants = program.get_program_body().split("NumberConstant(Number(")
    for i in range(program.get_program_body().count("NumberConstant")):
        number_constant = number_constants[i + 1].split(")")[0]    # (i + 1)th substring contains "x))())..."
        constants_representation += sspspace.bind(
            PROGRAM_NUMBER_SSP_SPACES[seed].encode(eval(number_constant)),    # Calling `eval` on string "Number(x)" evaluates to integer `x`
            PROGRAM_SLOT_SP_SPACES[seed].encode(np.array(["NUMBER"])),
        )

    return constants_representation
###############################################################################


# DSL SETUP
###############################################################################
# Define the data types in the DSL
TYPES = {
    MyObject,
    Colour,
    Number,
    Boolean,
    Hypothesis,
}

# Define the primitives in the DSL
PRIMITIVES = {
    ArgMax,
    ArgMin,
    Crop,
    Encircle,
    FillHoles,
    FillLines,
    Filter,
    HFlip,
    HStack,
    Hollow,
    PixelwiseAnd,
    PixelwiseNot,
    PixelwiseOr,
    PixelwiseXor,
    Recolour,
    Rectangify,
    Rot90,
    Rot180,
    Rot270,
    Scale,
    Transpose,
    VFlip,
    VStack,
    Xify,

    GridHypothesis,
    ColourHypothesis,
    TwoConnectedHypothesis,
    OneConnectedHypothesis,
    PartitionHypothesis,

    ColourConstant,
    ObjectGetColour,

    NumberConstant,
    ObjectGetColourCount,
    ObjectGetHeight,
    ObjectGetPixelCount,
    ObjectGetWidth,

    ObjectHasHoles,
    ObjectIsHSymmetrical,
    ObjectIsVSymmetrical,
}


# Define structured ways to represent DSL
N_PRIMITIVES = len(PRIMITIVES)
SORTED_PRIMITIVES = sorted(PRIMITIVES, key=(lambda x: x.__name__))
N_TYPES = len(TYPES)
SORTED_TYPES = sorted(TYPES, key=(lambda x: x.__name__))
PRIMITIVES_BY_TYPE = {t: [p for p in PRIMITIVES if (p.get_return_type() == t)] for t in TYPES}


# Create all possible programs
def enumerate_dsl(max_depth):
    atoms = {
        MyObject: {
            "x": [],
        },
        Colour: {
            "ColourConstant()()": [ColourConstant],
            "ObjectGetColour(x)()": [ObjectGetColour],
        },
        Number: {
            "NumberConstant()()": [NumberConstant],
            "ObjectGetColourCount(x)()": [ObjectGetColourCount],
            "ObjectGetHeight(x)()": [ObjectGetHeight],
            "ObjectGetPixelCount(x)()": [ObjectGetPixelCount],
            "ObjectGetWidth(x)()": [ObjectGetWidth],
        },
        Boolean: {
            "ObjectHasHoles(x)()": [ObjectHasHoles],
            "ObjectIsHSymmetrical(x)()": [ObjectIsHSymmetrical],
            "ObjectIsVSymmetrical(x)()": [ObjectIsVSymmetrical],
        },
    }

    for _ in range(max_depth):
        new_atoms = {}
        for p in PRIMITIVES_BY_TYPE[MyObject]:
            new_parameters = list(itertools.product(*(atoms[parameter_type] for parameter_type in p.get_parameter_type())))
            for new_parameter in new_parameters:
                primitives_used = [atoms[p.get_parameter_type()[i]][new_parameter[i]] for i in range(len(p.get_parameter_type()))]
                primitives_used = [primitives for subtree in primitives_used for primitives in subtree if primitives != []]
                new_atoms[f"{p.__name__}({', '.join(new_parameter)})()"] = [p] + primitives_used
        atoms[MyObject] |= new_atoms

    programs = []
    for program in list(atoms[MyObject].items()):
        for hypothesis in PRIMITIVES_BY_TYPE[Hypothesis]:
            programs.append((f"{hypothesis.__name__}; {program[0]}", [hypothesis] + program[1]))
    programs = sorted(programs, key=(lambda x: (len(x[1]), x[0])))

    return programs


# Define structured ways to represent programs in DSL
MAX_DEPTH = 2
NAIVE_PROGRAMS = [MyProgram(p[0], p[1]) for p in enumerate_dsl(MAX_DEPTH) if (len(p[1]) > 1)]    # If empty programs are desired replace with "enumerate_dsl(2)"
NAIVE_PROGRAMS_LENGTHS = [len(p.get_primitives()) for p in NAIVE_PROGRAMS]
NAIVE_PROGRAMS_PER_LENGTH = [NAIVE_PROGRAMS_LENGTHS.count(l) for l in range(max(NAIVE_PROGRAMS_LENGTHS) + 1)]
NAIVE_PROGRAMS_UPTO_LENGTH = [0]
EXPECTED_COST = {i: 0 for i in range(max(NAIVE_PROGRAMS_LENGTHS) + 1)}
for i in range(len(NAIVE_PROGRAMS_PER_LENGTH)):
    NAIVE_PROGRAMS_UPTO_LENGTH.append(NAIVE_PROGRAMS_UPTO_LENGTH[-1] + NAIVE_PROGRAMS_PER_LENGTH[i])
    EXPECTED_COST[i] = (NAIVE_PROGRAMS_UPTO_LENGTH[-1] + NAIVE_PROGRAMS_UPTO_LENGTH[-2]) / 2
NAIVE_PROGRAMS_UPTO_LENGTH.pop(0)
###############################################################################


# RESULTS VISUALIZATION SETUP
###############################################################################
# Plot the cumulative number of tasks solved
def plot_task_solving_efficiency(results, labels=None, plotter=plt):
    results = results.copy()
    scores = {}
    if (labels is None):
        labels = {l: l for l in results.keys()}
    elif (isinstance(labels, list)):
        labels = {l: l for l in labels}

    print("Scores (Average Area Under the Curve):")

    for i, l in enumerate(labels):
        logx = np.logspace(0, np.log10(len(NAIVE_PROGRAMS)), 100)
        precise_logx = np.logspace(0, np.log10(len(NAIVE_PROGRAMS)), 1000)

        cumulative_results = np.array([np.sum(np.logical_and(results[l] > 0, results[l] < x), axis=0) for x in logx])
        precise_cumulative_results = np.array([np.sum(np.logical_and(results[l] > 0, results[l] < x), axis=0) for x in precise_logx])

        mean_cumulative_results = np.mean(cumulative_results, axis=1)
        mean_precise_cumulative_results = np.mean(precise_cumulative_results, axis=1)
        min_cumulative_results = np.min(cumulative_results, axis=1)
        max_cumulative_results = np.max(cumulative_results, axis=1)

        scores[l] = np.trapezoid(mean_precise_cumulative_results.T, np.log10(precise_logx)) / np.trapezoid(np.full((len(precise_logx)), len(TASK_NAMES)), np.log10(precise_logx))
        print(f"\t{l.rjust(max([len(l) for l in results.keys()]))}: {scores[l]:.3f}")

        if (l == "Naive Baseline"):
            plotter.plot(logx, mean_cumulative_results, label=labels[l], color="#AAAAAA")
        else:
            plotter.plot(logx, mean_cumulative_results, label=labels[l])
            plotter.fill_between(logx, min_cumulative_results, max_cumulative_results, alpha=0.5)

    print()
    plotter.xscale("log")
    plotter.xlabel("Solutions Considered")
    plotter.ylim([len(TASK_NAMES) * -0.05, len(TASK_NAMES) * 1.05])
    plotter.ylabel("Tasks Solved")
    plotter.legend(loc="lower right")
    plotter.title("Task-Solving Efficiency")
    plotter.show()
    plotter.close()

    return scores


# Plot the number of attempts required to solve each task
def plot_task_by_task_results(results, labels=None):
    results = results.copy()
    if (labels is None):
        labels = results.keys()
    raw_results = np.flip(np.vstack([results[l].T for l in labels]), axis=1).T

    cutoffs = [0]
    for r in results.values():
        cutoffs.append(cutoffs[-1] + r.shape[1])

    processed_results = raw_results.copy()
    processed_results[processed_results == 0] = len(NAIVE_PROGRAMS)
    processed_results = np.log10(processed_results)
    processed_results = (processed_results - np.min(processed_results)) / (np.max(processed_results) - np.min(processed_results))

    plt.clf()
    plt.gcf().set_size_inches(processed_results.shape[1] / 2, processed_results.shape[0] / 4)

    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red", "black"])
    plt.pcolormesh(processed_results, cmap=custom_cmap)
    for i in range(raw_results.shape[0]):
        for j in range(raw_results.shape[1]):
            plt.text(j + 0.5, i + 0.5, f"{int(raw_results[i, j])}", ha="center", va="center", color="k")

    plt.xticks([(cutoffs[i] + cutoffs[i + 1]) / 2 for i in range(len(cutoffs) - 1)], results.keys())
    plt.xlabel("Solving Algorithm")
    plt.yticks(np.arange(len(TASK_NAMES)) + 0.5, [task_name.split("/")[-1].strip(".json") for task_name in TASK_NAMES[::-1]])
    plt.ylabel("Task ID")
    plt.title("Task-by-Task Sucess")
    for c in cutoffs[1:-1]:
        plt.axvline(c, color="w")
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    plt.show()
    plt.close()

    return
###############################################################################


# MISCELLANEOUS SETUP
###############################################################################
# Normalizes a VSA vector
def normalize(vector):
    if (np.all(vector == 0)):
        print("WARNING: Normalizing the zero vector")
        return vector
    return vector / np.linalg.norm(vector)


# Takes the cosine similarity between two VSA vectors
def cosine_similarity(representation_1, representation_2):
    representation_1 = np.atleast_2d(representation_1)
    representation_2 = np.atleast_2d(representation_2)
    assert (len(representation_1.shape) > 1) and (representation_1.shape[0] == 1) and (len(representation_2.shape) > 1) and (representation_2.shape[0] == 1) and (representation_1.shape == representation_2.shape), f"Cannot take similarity of {representation_1} and {representation_2}"
    if ((np.linalg.norm(representation_1) == 0) or (np.linalg.norm(representation_2) == 0)):
        return 0
    return (normalize(representation_1) @ normalize(representation_2).T)[0][0]


#
SOFTPLUS_BETA = 100
def softplus(x, bounded=True, temperature=SOFTPLUS_BETA):
    if (bounded):
        return np.log(1 + np.exp(temperature * x)) / temperature - np.log(1 + np.exp(temperature * (x - 1))) / temperature
    else:
        return np.log(1 + np.exp(SOFTPLUS_BETA * x)) / SOFTPLUS_BETA


#
SOFTMAX_TEMPERATURE = 1
def softmax(x, logits, temperature=SOFTMAX_TEMPERATURE):
    return np.exp(x / temperature) / np.sum(np.exp(np.array(logits) / temperature))
###############################################################################


def main():
    return


if __name__ == "__main__":
    main()
