# Isaac Jofe, 2025


from dsl import *
import json


TASKS_PATH = "../tasks/arcagi1/training/"


# Solvers for 21 naive grid-operation-based tasks
###############################################################################
def solve_1cf80156(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Crop(x)()
    return DecodeObjects(y)


def solve_28bf18c6(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = HStack(
        Crop(x)(),
        Crop(x)(),
    )()
    return DecodeObjects(y)


def solve_3c9b0459(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Rot180(x)()
    return DecodeObjects(y)


def solve_4c4377d9(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = VStack(
        VFlip(x)(),
        x,
    )()
    return DecodeObjects(y)


def solve_6150a2bd(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Rot180(x)()
    return DecodeObjects(y)


def solve_67a3c6ac(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = HFlip(x)()
    return DecodeObjects(y)


def solve_68b16354(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = VFlip(x)()
    return DecodeObjects(y)


def solve_6d0aefbc(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = HStack(
        x,
        HFlip(x)(),
    )()
    return DecodeObjects(y)


def solve_6fa7a44f(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = VStack(
        x,
        VFlip(x)(),
    )()
    return DecodeObjects(y)


def solve_7468f01a(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Crop(
        HFlip(x)(),
    )()
    return DecodeObjects(y)


def solve_74dd1130(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Transpose(x)()
    return DecodeObjects(y)


def solve_8be77c9e(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = VStack(
        x,
        VFlip(x)(),
    )()
    return DecodeObjects(y)


def solve_9172f3a0(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Scale(
        x,
        NumberConstant(Number(3))(),
    )()
    return DecodeObjects(y)


def solve_9dfd6313(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Transpose(x)()
    return DecodeObjects(y)


def solve_a416b8f3(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = HStack(
        x,
        x,
    )()
    return DecodeObjects(y)


def solve_ac0a08a4(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Scale(
        x,
        ObjectGetPixelCount(x)(),
    )()
    return DecodeObjects(y)


def solve_b91ae062(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Scale(
        x,
        ObjectGetColourCount(x)(),
    )()
    return DecodeObjects(y)


def solve_c59eb873(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Scale(
        x,
        NumberConstant(Number(2))(),
    )()
    return DecodeObjects(y)


def solve_c9e6f938(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = HStack(
        x,
        HFlip(x)(),
    )()
    return DecodeObjects(y)


def solve_ed36ccf7(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Rot270(x)()
    return DecodeObjects(y)


def solve_f25fbde4(input_grid):
    x = EncodeObjects(input_grid, "GridHypothesis")
    y = Crop(
        Scale(
            x,
            NumberConstant(Number(2))(),
        )(),
    )()
    return DecodeObjects(y)
###############################################################################


# Solvers for 18 naive object-operation-based tasks
###############################################################################
def solve_00d62c1b(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillHoles(
        x,
        ColourConstant(Colour(YELLOW))(),
    )()
    return DecodeObjects(y)


def solve_22168020(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillLines(
        x,
        ObjectGetColour(x)(),
    )()
    return DecodeObjects(y)


def solve_22eb0ac0(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillLines(
        x,
        ObjectGetColour(x)(),
    )()
    return DecodeObjects(y)


def solve_253bf280(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillLines(
        x,
        ColourConstant(Colour(GREEN))(),
    )()
    return DecodeObjects(y)


def solve_3aa6fb7a(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Rectangify(
        x,
        ColourConstant(Colour(BLUE))(),
    )()
    return DecodeObjects(y)


def solve_4258a5f9(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Encircle(
        x,
        ColourConstant(Colour(BLUE))(),
    )()
    return DecodeObjects(y)


def solve_4347f46a(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Hollow(
        x,
    )()
    return DecodeObjects(y)


def solve_50cb2852(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = FillHoles(
        Hollow(x)(),
        ColourConstant(Colour(CYAN))(),
    )()
    return DecodeObjects(y)


def solve_56ff96f3(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = Rectangify(
        x,
        ObjectGetColour(x)(),
    )()
    return DecodeObjects(y)


def solve_60b61512(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Rectangify(
        x,
        ColourConstant(Colour(ORANGE))(),
    )()
    return DecodeObjects(y)


def solve_623ea044(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Xify(
        x,
        ObjectGetColour(x)(),
    )()
    return DecodeObjects(y)


def solve_6d75e8bb(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Rectangify(
        x,
        ColourConstant(Colour(RED))(),
    )()
    return DecodeObjects(y)


def solve_6f8cd79b(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Recolour(
        Hollow(x)(),
        ColourConstant(Colour(CYAN))(),
    )()
    return DecodeObjects(y)


def solve_a5313dff(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillHoles(
        x,
        ColourConstant(Colour(BLUE))(),
    )()
    return DecodeObjects(y)


def solve_bb43febb(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = FillHoles(
        Hollow(x)(),
        ColourConstant(Colour(RED))(),
    )()
    return DecodeObjects(y)


def solve_ce22a75a(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Recolour(
        Encircle(
            x,
            ObjectGetColour(x)(),
        )(),
        ColourConstant(Colour(BLUE))(),
    )()
    return DecodeObjects(y)


def solve_dbc1a6ce(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillLines(
        x,
        ColourConstant(Colour(CYAN))(),
    )()
    return DecodeObjects(y)


def solve_ded97339(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = FillLines(
        x,
        ObjectGetColour(x)(),
    )()
    return DecodeObjects(y)
###############################################################################


# Solvers for 6 naive object-combination-based tasks
###############################################################################
def solve_0520fde7(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = Recolour(
        PixelwiseAnd(
            x,
        )(),
        ColourConstant(Colour(RED))(),
    )()
    return DecodeObjects(y)


def solve_3428a4f5(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = Recolour(
        PixelwiseXor(
            x,
        )(),
        ColourConstant(Colour(GREEN))(),
    )()
    return DecodeObjects(y)


def solve_99b1bc43(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = Recolour(
        PixelwiseXor(
            x,
        )(),
        ColourConstant(Colour(GREEN))(),
    )()
    return DecodeObjects(y)


def solve_ce4f8723(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = Recolour(
        PixelwiseOr(
            x,
        )(),
        ColourConstant(Colour(GREEN))(),
    )()
    return DecodeObjects(y)


def solve_dae9d2b5(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = Recolour(
        PixelwiseOr(
            x,
        )(),
        ColourConstant(Colour(PINK))(),
    )()
    return DecodeObjects(y)


def solve_e98196ab(input_grid):
    x = EncodeObjects(input_grid, "PartitionHypothesis")
    y = PixelwiseOr(
        x,
    )()
    return DecodeObjects(y)
###############################################################################


# Solvers for 8 naive object-selection-based tasks
###############################################################################
def solve_0b148d64(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = Crop(
        ArgMin(
            x,
            ObjectGetPixelCount(x)(),
        )(),
    )()
    return DecodeObjects(y)


def solve_1f85a75f(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Crop(
        ArgMax(
            x,
            ObjectGetPixelCount(x)(),
        )(),
    )()
    return DecodeObjects(y)


def solve_23b5c85d(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Crop(
        ArgMin(
            x,
            ObjectGetPixelCount(x)(),
        )(),
    )()
    return DecodeObjects(y)


def solve_25d8a9c8(input_grid):
    x = EncodeObjects(input_grid, "OneConnectedHypothesis")
    y = Recolour(
        ArgMax(
            x,
            ObjectGetWidth(x)(),
        )(),
        ColourConstant(Colour(GREY))(),
    )()
    return DecodeObjects(y)


def solve_31aa019c(input_grid):
    x = EncodeObjects(input_grid, "ColourHypothesis")
    y = Encircle(
        ArgMin(
            x,
            ObjectGetPixelCount(x)(),
        )(),
        ColourConstant(Colour(RED))(),
    )()
    return DecodeObjects(y)


def solve_72ca375d(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Crop(
        Filter(
            x,
            ObjectIsHSymmetrical(x)(),
        )(),
    )()
    return DecodeObjects(y)


def solve_a87f7484(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Crop(
        ArgMax(
            x,
            ObjectGetPixelCount(x)(),
        )(),
    )()
    return DecodeObjects(y)


def solve_be94b721(input_grid):
    x = EncodeObjects(input_grid, "TwoConnectedHypothesis")
    y = Crop(
        ArgMax(
            x,
            ObjectGetPixelCount(x)(),
        )(),
    )()
    return DecodeObjects(y)
###############################################################################


def check_solvers(task_names):
    print(f"Verifying solvers for {len(task_names)} tasks...")
    print()

    n_tasks_correct = 0
    for task_name in task_names:
        # Load task data
        print(f"\tVerifying task {task_name}...")
        with open(f"{TASKS_PATH}{task_name}.json", "r") as f:
            task = json.load(f)

        # Test that the program solves all demonstrations
        n_demonstrations_correct = 0
        for i in range(len(task["train"])):
            guess = eval(f"solve_{task_name}(task['train'][i]['input'])")
            n_demonstrations_correct += (guess.shape == np.array(task["train"][i]["output"]).shape) and np.all(guess == np.array(task["train"][i]["output"]))
        print(f"\t\t{n_demonstrations_correct}/{len(task['train'])} demonstrations correct")

        # Test that the program solves all queries
        n_queries_correct = 0
        for i in range(len(task["test"])):
            guess = eval(f"solve_{task_name}(task['test'][i]['input'])")
            n_queries_correct += (guess.shape == np.array(task["test"][i]["output"]).shape) and np.all(guess == np.array(task["test"][i]["output"]))
        print(f"\t\t{n_queries_correct}/{len(task['test'])} queries correct")

        # Check whether all tests passed
        if (len(task["train"]) == n_demonstrations_correct) and (len(task["test"]) == n_queries_correct):
            n_tasks_correct += 1
            print(f"\tTask {task_name} verified.")
        else:
            print(f"\tTask {task_name} not verified.")
        print()

    print(f"Solvers for {n_tasks_correct}/{len(task_names)} tasks verified.")
    return


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


def main():
    check_solvers(TASK_NAMES)
    return


if __name__ == "__main__":
    main()
