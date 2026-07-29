# Isaac Joffe, 2026


from utils import *


# PROGRAM IMPLEMENTATION
###############################################################################
# Execute a program on an input
def execute_program(input_grid, program):
    x = EncodeObjects(input_grid, program.get_program_hypothesis())
    y = eval(program.get_program_body())    # Implicitly requires x
    z = DecodeObjects(y)
    return z


# Safely execute a program on an input
def safe_execute_program(input_grid, program):
    try:
        return execute_program(input_grid, program)

    except Exception as e:
        if (DEBUG):
            print(f"Exception thrown during execution of program `{program}`: {e}")
        return np.array([[]])


# Check if a program solves an input-output pair
def test_program(input_grid, program, output_grid):
    predicted_grid = safe_execute_program(input_grid, program)
    return (np.all(predicted_grid.shape == output_grid.shape) and np.all(predicted_grid == output_grid))


# Check if program solves all demonstrations
def test_demonstrations(demonstrations, program):
    train_ins, train_outs = demonstrations
    return np.all([test_program(train_in, program, train_out) for train_in, train_out in zip(train_ins, train_outs)])


# Check if program solves all queries
def test_queries(queries, program):
    test_ins, test_outs = queries
    return np.all([test_program(test_in, program, test_out) for test_in, test_out in zip(test_ins, test_outs)])


# Check if a program solves all of a task's queries
def check_solutions(queries, solution_programs, print_result=True):
    if (solution_programs):
        for i, (j, solution_program) in enumerate(solution_programs):
            if (test_queries(queries, solution_program)):
                if (print_result):
                    print(f"Correct: Found solution program {i + 1} correctly solved all queries.")
                return j, solution_program

        if (print_result):
            print(f"Incorrect: Found {len(solution_programs)} solution program(s), but all of them failed on at least 1 query.")
        return 0, get_null_program()

    if (print_result):
        print("Incorrect: Found 0 solution programs.")
    return 0, get_null_program()


# Solve all tasks
N_ATTEMPTS = 1
def try_all_tasks(metadata, results, labels=None):
    if (labels is None):
        labels = [l for l in metadata.keys() if (l not in results.keys())]
    print(f"Collecting results for: {', '.join(labels)}")

    for l in labels:
        print(f"\nNow using: {l}")
        results[l] = np.zeros((len(TASKS), len(metadata[l])))

        for i, task_name in enumerate(TASKS):
            print(f"\nNow solving {task_name.split('/')[-1].strip('.json')}...")

            demonstrations = get_demonstrations(TASKS[task_name])
            queries = get_queries(TASKS[task_name])

            for j in range(len(metadata[l])):

                if ("Naive Baseline" in l):
                    solution_programs = solve_task_by_enumeration(demonstrations, NAIVE_PROGRAMS)
                    results[l][i][j] += [EXPECTED_COST[len(check_solutions(queries, solution_programs)[1].get_primitives())]]

                else:
                    if ("DeepCoder-Style Enumeration" in l):
                        ranked_programs = deepcoder_inference_rank(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_enumeration(demonstrations, ranked_programs)

                    elif ("DeepCoder-Style Evolution" in l):
                        program_predictions = deepcoder_inference(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_evolution(demonstrations, program_predictions, mode="DeepCoder", **metadata[l][j])

                    elif ("Direct-VSA Enumeration" in l):
                        ranked_programs = vsa_inference_rank(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_enumeration(demonstrations, ranked_programs)

                    elif ("Direct-VSA Evolution" in l):
                        program_predictions = vsa_inference(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_evolution(demonstrations, program_predictions, mode="Direct", **metadata[l][j])

                    elif ("Hybrid-VSA Enumeration" in l):
                        ranked_programs = hybrid_inference_rank(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_enumeration(demonstrations, ranked_programs)

                    elif ("Hybrid-DeepCoder Evolution" in l):
                        program_predictions = deepcoder_inference(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_evolution(demonstrations, program_predictions, mode="Hybrid-DeepCoder", **metadata[l][j])

                    elif ("Hybrid-VSA Evolution" in l):
                        program_predictions = hybrid_inference(demonstrations, **metadata[l][j])
                        solution_programs = solve_task_by_evolution(demonstrations, program_predictions, mode="Hybrid-VSA", **metadata[l][j])

                    elif ("GOAT" in l):
                        solution_programs = solve_task_by_GOAT(demonstrations, **metadata[l][j])

                    else:
                        assert False, f"Invalid label \"{l}\": No corresponding algorithm"

                    results[l][i][j] += [check_solutions(queries, solution_programs)[0]]

    return results
###############################################################################


# PROGRAM CONSTANT IMPLEMENTATION
###############################################################################
COLOUR_CONSTANTS = [
    "Colour(BLACK)",
    "Colour(BLUE)",
    "Colour(RED)",
    "Colour(GREEN)",
    "Colour(YELLOW)",
    "Colour(GREY)",
    "Colour(PINK)",
    "Colour(ORANGE)",
    "Colour(CYAN)",
    "Colour(PURPLE)",
]

NUMBER_CONSTANTS = [
    "Number(1)",
    "Number(2)",
    "Number(3)",
    "Number(4)",
    "Number(5)",
    # "Number(6)",
    # "Number(7)",
    # "Number(8)",
    # "Number(9)",
    # "Number(10)",
]


def assign_constants_to_program(program, predict_constants=False, constant_representation=None):
    if (predict_constants):
        assert (constant_representation is not None), "To predict constant values, a VSA representation must be supplied"

        colour_prediction = sspspace.bind(
            constant_representation,
            sspspace.invert(PROGRAM_SLOT_SP_SPACE.encode(np.array(["COLOUR"]))),
        )
        colour_constant_similarities = [(colour_constant, np.dot(colour_prediction, PROGRAM_COLOUR_SP_SPACE.encode(np.array([colour_constant.split("(")[1].split(")")[0]])))) for colour_constant in COLOUR_CONSTANTS]
        number_prediction = sspspace.bind(
            constant_representation,
            sspspace.invert(PROGRAM_SLOT_SP_SPACE.encode(np.array(["NUMBER"]))),
        )
        number_constant_similarities = [(number_constant, np.dot(number_prediction, PROGRAM_NUMBER_SSP_SPACE.encode(number_constant.split("(")[1].split(")")[0]))) for number_constant in NUMBER_CONSTANTS]

    else:
        colour_constant_similarities = [(colour_constant, 1) for colour_constant in COLOUR_CONSTANTS]
        number_constant_similarities = [(number_constant, 1) for number_constant in NUMBER_CONSTANTS]

    possible_constants = []
    for i in range(program.get_program().count("Constant()")):
        if (program.get_program().split("Constant()")[i].endswith("Colour")):
            possible_constants.append(colour_constant_similarities)
        elif (program.get_program().split("Constant()")[i].endswith("Number")):
            possible_constants.append(number_constant_similarities)
        else:
            assert False, f"Invalid constant in {program.get_program()}"

    possible_programs = []
    for constant_assignment in itertools.product(*possible_constants):
        possible_program = program.get_program()
        for constant, _ in constant_assignment:
            index = possible_program.find("Constant()")
            possible_program = possible_program[:index] + f"Constant({constant})" + possible_program[(index + 10):]
        possible_programs.append((MyProgram(possible_program, program.get_primitives()), np.prod([score for _, score in constant_assignment])))

    # Filter out porgrams that have highly improbable constant configurations
    possible_programs = [program for program, score in sorted(possible_programs, key=(lambda x: -x[1])) if (score >= 0.1)]

    return possible_programs
###############################################################################


# UTILITIES FOR PROGRAM RANKING ALGORITHMS
###############################################################################
def reduce_probabilities(probabilities, reduction, axis):
    assert (probabilities.shape[-1] == N_PRIMITIVES), f"Error in dimensions \"{probabilities.shape}\" before reduction over probabilities \"{reduction}\""
    if (reduction.lower() == "mean"):
        probabilities = np.mean(probabilities, axis=axis)
    elif (reduction.lower() == "min"):
        probabilities = np.min(probabilities, axis=axis)
    elif (reduction.lower() == "max"):
        probabilities = np.max(probabilities, axis=axis)
    else:
        assert False, f"Invalid reduction over probabilities \"{reduction}\""
    assert (probabilities.shape[-1] == N_PRIMITIVES), f"Error in dimensions \"{probabilities.shape}\" after reduction over probabilities \"{reduction}\""
    return probabilities


def reduce_vsa(representations, reduction, axis):
    assert (representations.shape[-1] == PROGRAM_REPRESENTATION_DIMENSIONS), f"Error in dimensions \"{representations.shape}\" before reduction over VSA representations \"{reduction}\""
    if (reduction.lower() == "mean"):
        representations = np.mean(representations, axis=axis)
    else:
        assert False, f"Invalid reduction over VSA representations \"{reduction}\""
    assert (representations.shape[-1] == PROGRAM_REPRESENTATION_DIMENSIONS), f"Error in dimensions \"{representations.shape}\" after reduction over VSA representations \"{reduction}\""
    return representations


def reduce_programs(programs, reduction):
    assert ((len(programs) > 0) and (all((len(programs[0]) == len(programs[i])) for i in range(1, len(programs))))), f"Error in dimensions \"{(len(programs), len(programs[0]))}\" before reduction over programs \"{reduction}\""
    scored_programs = []
    if (reduction.lower() == "mean"):
        for j in range(len(NAIVE_PROGRAMS)):
            score = np.mean([programs[i][j].get_score() for i in range(len(programs))])
            scored_programs.append(programs[0][j].copy_with_score(score=score))
    elif (reduction.lower() == "min"):
        for j in range(len(NAIVE_PROGRAMS)):
            index = np.argmin([programs[i][j].get_score() for i in range(len(programs))])
            scored_programs.append(programs[index][j])
    elif (reduction.lower() == "max"):
        for j in range(len(NAIVE_PROGRAMS)):
            index = np.argmax([programs[i][j].get_score() for i in range(len(programs))])
            scored_programs.append(programs[index][j])
    else:
        assert False, f"Invalid reduction over programs \"{reduction}\""
    return scored_programs


def project_probabilities_vsa(probabilities, embeddings, projection):
    assert (probabilities.shape[-1] == N_PRIMITIVES), f"Error in dimensions \"{probabilities.shape}\" before reduction over probabilities \"{reduction}\""
    if (projection.lower() == "regression"):
        projections = np.linalg.lstsq(embeddings, probabilities, rcond=None)[0]
    elif (projection.lower() == "matrix"):
        projections = probabilities @ embeddings
    else:
        assert False, f"Invalid projection from probabilities to VSA \"{projection}\""
    assert (projections.shape[-1] == PROGRAM_REPRESENTATION_DIMENSIONS), f"Error in dimensions \"{projections.shape}\" after projection from probabilities to VSA representation \"{projection}\""
    return projections


def reduce_scores(scores, reduction):
    if (reduction.lower() == "mean"):
        score = np.mean(scores)
    elif (reduction.lower() == "min"):
        score = np.min(scores)
    elif (reduction.lower() == "max"):
        score = np.max(scores)
    elif (reduction.lower() == "mean-max"):
        score = np.mean(scores) + np.max(scores)
    else:
        assert False, f"Invalid reduction over scores \"{reduction}\""
    return score


def sort_programs(programs):
    return sorted(programs, key=(lambda x: -x.get_score()))


def print_probabilities(probabilities, preface):
    print(f"{preface}, primitive-by-primitive probabilities:")
    print(*[f"{p.__name__}={probabilities[i]:.2f}" for i, p in enumerate(SORTED_PRIMITIVES)], sep=", ")
    return


def print_similarities(similarities, embeddings, preface):
    print(f"{preface}, primitive-by-primitive similarities:")
    print(*[f"{p.__name__}={np.dot(similarities[i], embeddings[i]):.2f}" for i, p in enumerate(SORTED_PRIMITIVES)], sep=", ")
    return


def print_programs(programs, preface, n=10):
    print(f"{preface}, top candidate programs:",
        *[f"`{program}` [{program.get_score():.3f}]" for program in programs[:n]], sep="\n\t")
    return
###############################################################################


# DEEPCODER-BASED GUIDANCE
###############################################################################
def deepcoder_score(program, primitive_predictions):
    return np.prod([primitive_predictions[SORTED_PRIMITIVES.index(p)] for p in program.get_primitives()])


def deepcoder_inference(
    demonstrations,
    prediction_nns,
    demonstration_reduction,    # "Mean"
    predict_constants=False,
    **kwargs,    # Allow calls with superfluous arguments
):
    primitive_predictions = np.zeros((len(prediction_nns), len(demonstrations[0]), N_PRIMITIVES))
    if (predict_constants):
        constant_predictions = np.zeros((len(prediction_nns), len(demonstrations[0]), PROGRAM_REPRESENTATION_DIMENSIONS))

    for i, prediction_nn in enumerate(prediction_nns):
        for j, (train_in, train_out) in enumerate(zip(*demonstrations)):
            if (predict_constants):
                primitive_predictions[i][j], constant_predictions[i][j] = prediction_nn.predict(train_in, train_out)
            else:
                primitive_predictions[i][j] = prediction_nn.predict(train_in, train_out)

    # Reduce over demonstrations
    reduced_primitive_predictions = reduce_probabilities(primitive_predictions, demonstration_reduction, axis=1)
    if (predict_constants):
        reduced_constant_predictions = reduce_vsa(constant_predictions, "Mean", axis=1)

    assert (reduced_primitive_predictions.shape == (len(prediction_nns), N_PRIMITIVES)), f"Invalid prediction shape \"{primitive_predictions.shape}\""
    if (predict_constants):
        assert (reduced_constant_predictions.shape == (len(prediction_nns), PROGRAM_REPRESENTATION_DIMENSIONS)), f"Invalid prediction shape \"{constant_predictions.shape}\""

    if (predict_constants):
        return reduced_primitive_predictions, reduced_constant_predictions
    else:
        return reduced_primitive_predictions


def deepcoder_rank(
    predictions,
    predictor_reduction,    # "Mean-Score",
    predict_constants=False,
    constant_representation=None,
    **kwargs,    # Allow calls with superfluous arguments
):
    if (DEBUG):
        for i in range(len(predictions)):
            print_probabilities(predictions[i], f"Predictor {i}")

    # Perform score-then-reduce (mean, min, or max) strategy
    if (predictor_reduction.lower().split("-")[0] == "score"):

        # First, score programs according to separate predictions
        scored_programs = []
        for i in range(len(predictions)):
            scored_programs.append([
                program.copy_with_score(
                    score=deepcoder_score(program, predictions[i]),
                ) for program in sorted(NAIVE_PROGRAMS, key=(lambda x: x.get_program()))
            ])
            if (DEBUG):
                print_programs(sort_programs(ranked_programs), f"Predictor {i}")

        # Second, reduce program scores to form a single ranking
        reduced_scored_programs = reduce_programs(scored_programs, predictor_reduction.split("-")[1])

    # Perform reduce (mean, min, or max) then-score strategy
    elif (predictor_reduction.lower().split("-")[1] == "score"):

        # First, reduce separate predictions into a single prediction
        reduced_predictions = reduce_probabilities(predictions, predictor_reduction.split("-")[0], axis=0)
        if (DEBUG):
            print_probabilities(reduced_predictions, "Reduced")

        # Second, score programs based on the reduced single prediction
        reduced_scored_programs = [
            program.copy_with_score(
                score=deepcoder_score(program, reduced_predictions),
            ) for program in NAIVE_PROGRAMS
        ]

    else:
        assert False, f"Invalid reduction over the predictors \"{predictor_reduction}\""

    # Third, reduce over constants, if needed
    if (predict_constants):
        reduced_constant_predictions = reduce_vsa(constant_representation, "Mean", axis=0)

    # Fourth, sort programs according to their single reduced score
    ranked_programs = sort_programs(reduced_scored_programs)
    if (DEBUG):
        print_programs(ranked_programs, "Reduced")

    if (predict_constants):
        return ranked_programs, reduced_constant_predictions
    else:
        return ranked_programs


def deepcoder_inference_rank(
    demonstrations,
    prediction_nns,
    predictor_reduction,    # "Mean-Score",
    demonstration_reduction,    # "Mean",
    thumbs=None,
    predict_constants=False,
    **kwargs,    # Allow calls with superfluous arguments
):
    if (predict_constants):
        primitive_predictions, constant_predictions = deepcoder_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
            predict_constants=True,
        )

        if (thumbs is not None):
            for p in thumbs:
                primitive_predictions[:, SORTED_PRIMITIVES.index(p)] = THUMBS_VALUE

        ranked_programs, reduced_constant_predictions = deepcoder_rank(
            primitive_predictions,
            predictor_reduction=predictor_reduction,
            predict_constants=True,
            constant_predictions=constant_predictions,
        )
        return ranked_programs, reduced_constant_predictions

    else:
        primitive_predictions = deepcoder_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
        )

        if (thumbs is not None):
            for p in thumbs:
                primitive_predictions[:, SORTED_PRIMITIVES.index(p)] = THUMBS_VALUE

        ranked_programs = deepcoder_rank(
            primitive_predictions,
            predictor_reduction=predictor_reduction,
        )
        return ranked_programs
###############################################################################


# VSA-BASED GUIDANCE
###############################################################################
def vsa_score(
    level,
    primitive,
    representation,
    probability,
    embeddings,
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    **kwargs,
):
    # Search for object hypothesis and top-level primitive
    if (level == 0):
        possible_trees = []

        for hypothesis in sorted(PRIMITIVES_BY_TYPE[Hypothesis], key=(lambda x: -np.dot(representation, embeddings[x])))[:beam_size]:
            if (score_mode.lower() == "softplus"):
                hypothesis_probability = softplus(np.dot(representation, embeddings[hypothesis]))
            elif (score_mode.lower() == "softmax"):
                hypothesis_probability = softmax(np.dot(representation, embeddings[hypothesis]), [np.dot(representation, embeddings[p]) for p in PRIMITIVES_BY_TYPE[Hypothesis]])
            else:
                hypothesis_probability = np.dot(representation, embeddings[hypothesis])

            if (subtract_mode.lower() == "scaled"):
                hypothesis_representation = representation - hypothesis_probability * embeddings[hypothesis]
            elif (subtract_mode.lower() == "no subtract"):
                hypothesis_representation = representation
            else:
                hypothesis_representation = representation - embeddings[hypothesis]

            possible_subtrees = []
            for next_primitive in sorted(PRIMITIVES_BY_TYPE[MyObject], key=(lambda x: -np.dot(hypothesis_representation, embeddings[x])))[:beam_size]:
                if (score_mode.lower() == "softplus"):
                    next_probability = softplus(np.dot(hypothesis_representation, embeddings[next_primitive]))
                elif (score_mode.lower() == "softmax"):
                    next_probability = softmax(np.dot(hypothesis_representation, embeddings[next_primitive]), [np.dot(hypothesis_representation, embeddings[p]) for p in PRIMITIVES_BY_TYPE[MyObject]])
                else:
                    next_probability = np.dot(hypothesis_representation, embeddings[next_primitive])

                if (subtract_mode.lower() == "scaled"):
                    next_representation = hypothesis_representation - next_probability * embeddings[next_primitive]
                elif (subtract_mode.lower() == "no subtract"):
                    next_representation = hypothesis_representation
                else:
                    next_representation = hypothesis_representation - embeddings[next_primitive]

                possible_subtrees += vsa_score(
                    level + 1,
                    next_primitive,
                    next_representation,
                    next_probability,
                    embeddings=embeddings,
                    score_mode=score_mode,
                    subtract_mode=subtract_mode,
                    beam_size=beam_size,
                    max_depth=max_depth,
                )

            for possible_subtree in possible_subtrees:
                possible_trees.append(
                    MyProgram(
                        f"{hypothesis.__name__}; {possible_subtree[0]}",
                        [hypothesis] + possible_subtree[1],
                        hypothesis_probability * possible_subtree[2],
                    )
                )

        return possible_trees

    # Search for possible subtrees for each parameter of this primitive
    possible_subtrees = []
    for parameter_type in primitive.get_parameter_type():

        # Parameter is objects (for which (capped) deeper search is necessary)
        if (parameter_type == MyObject):
            possible_subtrees.append([("x", [], 1)])
            if (level < max_depth):
                for next_primitive in sorted(PRIMITIVES_BY_TYPE[parameter_type], key=(lambda x: -np.dot(representation, embeddings[x])))[:beam_size]:
                    if (score_mode.lower() == "softplus"):
                        next_probability = softplus(np.dot(representation, embeddings[next_primitive]))
                    elif (score_mode.lower() == "softmax"):
                        next_probability = softmax(np.dot(representation, embeddings[next_primitive]), [np.dot(representation, embeddings[p]) for p in PRIMITIVES_BY_TYPE[parameter_type]])
                    else:
                        next_probability = np.dot(representation, embeddings[next_primitive])

                    if (subtract_mode.lower() == "scaled"):
                        next_representation = representation - next_probability * embeddings[next_primitive]
                    elif (subtract_mode.lower() == "no subtract"):
                        next_representation = representation
                    else:
                        next_representation = representation - embeddings[next_primitive]

                    possible_subtrees[-1] += vsa_score(
                        level + 1,
                        next_primitive,
                        next_representation,
                        next_probability,
                        embeddings=embeddings,
                        score_mode=score_mode,
                        subtract_mode=subtract_mode,
                        beam_size=beam_size,
                        max_depth=max_depth,
                    )

        # Parameter is of another type (for which deeper search is not necessary)
        else:
            possible_subtrees.append([])
            for next_primitive in sorted(PRIMITIVES_BY_TYPE[parameter_type], key=(lambda x: -np.dot(representation, embeddings[x])))[:beam_size]:
                if (score_mode.lower() == "softplus"):
                    next_probability = softplus(np.dot(representation, embeddings[next_primitive]))
                elif (score_mode.lower() == "softmax"):
                    next_probability = softmax(np.dot(representation, embeddings[next_primitive]), [np.dot(representation, embeddings[p]) for p in PRIMITIVES_BY_TYPE[parameter_type]])
                else:
                    next_probability = np.dot(representation, embeddings[next_primitive])

                possible_subtrees[-1] += [(f"{next_primitive.__name__}({'x' if (next_primitive.get_parameter_type() == [MyObject]) else ''})()", [next_primitive], next_probability)]

    # Combine possible subtrees into the trees for this primitive
    possible_trees = []
    for possible_subtree in itertools.product(*possible_subtrees):
        possible_trees.append((f"{primitive.__name__}({', '.join([s[0] for s in possible_subtree])})()", [primitive] + list(itertools.chain(*[s[1] for s in possible_subtree])), probability * np.prod([s[2] for s in possible_subtree])))

    return possible_trees


def vsa_inference(
    demonstrations,
    prediction_nns,
    demonstration_reduction,    # "Mean"
    predict_constants=False,
    **kwargs,    # Allow calls with superfluous arguments
):
    primitive_predictions = np.zeros((len(prediction_nns), len(demonstrations[0]), PROGRAM_REPRESENTATION_DIMENSIONS))
    if (predict_constants):
        constant_predictions = np.zeros((len(prediction_nns), len(demonstrations[0]), PROGRAM_REPRESENTATION_DIMENSIONS))

    for i, prediction_nn in enumerate(prediction_nns):
        for j, (train_in, train_out) in enumerate(zip(*demonstrations)):
            if (predict_constants):
                primitive_predictions[i][j], constant_predictions[i][j] = prediction_nn.predict(train_in, train_out)
            else:
                primitive_predictions[i][j] = prediction_nn.predict(train_in, train_out)

    # Reduce over demonstrations
    reduced_primitive_predictions = reduce_vsa(primitive_predictions, demonstration_reduction, axis=1)
    if (predict_constants):
        reduced_constant_predictions = reduce_vsa(constant_predictions, "Mean", axis=1)

    assert (reduced_primitive_predictions.shape == (len(prediction_nns), PROGRAM_REPRESENTATION_DIMENSIONS)), f"Invalid prediction shape \"{primitive_predictions.shape}\""
    if (predict_constants):
        assert (reduced_constant_predictions.shape == (len(prediction_nns), PROGRAM_REPRESENTATION_DIMENSIONS)), f"Invalid prediction shape \"{constant_predictions.shape}\""

    if (predict_constants):
        return reduced_primitive_predictions, reduced_constant_predictions
    else:
        return reduced_primitive_predictions


def vsa_rank(
    predictions,
    embeddings,
    predictor_reduction,    # "Mean-Score",
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    predict_constants=False,
    constant_representation=None,
    **kwargs,
):
    if (DEBUG):
        for i in range(len(predictions)):
            print_similarities(predictions[i], embeddings, f"Predictor {i}")

    # Perform score-then-reduce (mean, min, or max) strategy
    if (predictor_reduction.lower().split("-")[0] == "score"):

        # First, score programs according to separate predictions
        scored_programs = []
        for i in range(len(predictions)):
            scored_programs.append(
                sorted(
                    vsa_score(
                        level=0,
                        primitive=None,
                        representation=predictions[i],
                        probability=1,
                        embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
                        score_mode=score_mode,
                        subtract_mode=subtract_mode,
                        beam_size=beam_size,
                        max_depth=max_depth,
                        **kwargs,
                    ),
                    key=(lambda x: x.get_program()),
                )
            )
            if (DEBUG):
                print_programs(sort_programs(scored_programs[-1]), f"Predictor {i}")

        # Second, reduce program scores to form a single ranking
        reduced_scored_programs = reduce_programs(scored_programs, predictor_reduction.split("-")[1])

    # Perform reduce (mean) then-score strategy
    elif (predictor_reduction.lower().split("-")[1] == "score"):

        # First, reduce separate predictions into a single prediction
        reduced_predictions = reduce_vsa(predictions, predictor_reduction.split("-")[0], axis=0)
        if (DEBUG):
            print_similarities(reduced_predictions, embeddings, "Reduced")

        # Second, score programs based on the reduced single prediction
        reduced_scored_programs = vsa_score(
            level=0,
            primitive=None,
            representation=reduced_predictions,
            probability=1,
            embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            **kwargs,
        )

    else:
        assert False, f"Invalid reduction over the predictors \"{predictor_reduction}\""

    # Third, reduce over constants, if needed
    if (predict_constants):
        reduced_constant_predictions = reduce_vsa(constant_representation, "Mean", axis=0)

    # Fourth, sort programs according to their single reduced score
    ranked_programs = sort_programs(reduced_scored_programs)
    if (DEBUG):
        print_programs(ranked_programs, "Reduced")

    if (predict_constants):
        return ranked_programs, reduced_constant_predictions
    else:
        return ranked_programs


def vsa_inference_rank(
    demonstrations,
    prediction_nns,
    embeddings,
    predictor_reduction,    # "Mean-Score"
    demonstration_reduction,    # "Mean"
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    predict_constants=False,
    **kwargs,
):
    if (predict_constants):
        primitive_predictions, constant_predictions = vsa_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
            predict_constants=True,
        )
        ranked_programs, reduced_constant_predictions = vsa_rank(
            primitive_predictions,
            embeddings=embeddings,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            predict_constants=True,
            constant_predictions=constant_predictions,
            **kwargs,
        )
        return ranked_programs, reduced_constant_predictions

    else:
        primitive_predictions = vsa_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
        )
        ranked_programs = vsa_rank(
            primitive_predictions,
            embeddings=embeddings,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            **kwargs,
        )
        return ranked_programs
###############################################################################


# COMBINATION OF DEEPCDOER AND VSA ELEMENTS
###############################################################################
def hybrid_inference(
    demonstrations,
    prediction_nns,
    embeddings,
    projection_mode,    # "Matrix",
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    predictor_reduction,    # "Mean-Score",
    demonstration_reduction,    # "Mean",
    thumbs=None,
    **kwargs,
):
    primitive_predictions = deepcoder_inference(
        demonstrations=demonstrations,
        prediction_nns=prediction_nns,
        demonstration_reduction=demonstration_reduction,
    )

    if (thumbs is not None):
        for p in thumbs:
            primitive_predictions[:, SORTED_PRIMITIVES.index(p)] = THUMBS_VALUE

    # Perform project-then-reduce (mean, min, or max) strategy
    if (predictor_reduction.lower().split("-")[0] == "project"):

        # First, project separate predictions into separate representations
        projected_predictions = project_probabilities_vsa(primitive_predictions[i], embeddings, projection_mode)

        # Second, reduce program scores to form single ranking
        projected_predictions = reduce_vsa(projected_predictions, predictor_reduction.split("-")[0], axis=0)

    # Perform reduce (mean, min, or max) then-project strategy
    elif (predictor_reduction.lower().split("-")[1] == "project"):

        # First, reduce separate predictions into single predictions
        primitive_predictions = reduce_probabilities(primitive_predictions, predictor_reduction.split("-")[0], axis=0)
        if (DEBUG):
            print_probabilities(primitive_predictions, "Reduced")

        # Second, score programs based on reduced predictions
        projected_predictions = project_probabilities_vsa(primitive_predictions, embeddings, projection_mode)

    else:
        assert False, f"Invalid reduction over the predictors \"{predictor_reduction}\""

    return projected_predictions


def hybrid_rank(
    predictions,
    embeddings,
    projection_mode,    # "Matrix",
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    predictor_reduction,    # "Mean-Score",
    predict_constants=False,
    constant_representation=None,
    **kwargs,
):
    if (DEBUG):
        for i in range(len(predictions)):
            print_probabilities(predictions[i], f"Predictor {i}")

    # Perform project-then-score-then-reduce (mean, min, or max) strategy
    if ((predictor_reduction.lower().split("-")[0] == "project") and (predictor_reduction.lower().split("-")[1] == "score")):
        scored_programs = []
        for i in range(len(predictions)):

            # First, project separate predictions into separate representations
            representations = project_probabilities_vsa(predictions[i], embeddings, projection_mode)
            if (DEBUG):
                print_similarities(representations[i], embeddings, f"Predictor {i}")

            # Second, score programs based on the separate representations
            scored_programs.append(
                sorted(
                    vsa_score(
                        level=0,
                        primitive=None,
                        representation=representations,
                        probability=1,
                        embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
                        score_mode=score_mode,
                        subtract_mode=subtract_mode,
                        beam_size=beam_size,
                        max_depth=max_depth,
                        **kwargs,
                    ),
                    key=(lambda x: x.get_program()),
                )
            )
            if (DEBUG):
                print_programs(sort_programs(scored_programs[-1]), f"Predictor {i}")

        # Third, reduce separate program scores to form a single ranking
        reduced_scored_programs = reduce_programs(scored_programs, predictor_reduction.split("-")[2])

    # Perform project-then-reduce (mean) then-score strategy
    elif ((predictor_reduction.lower().split("-")[0] == "project") and (predictor_reduction.lower().split("-")[2] == "score")):

        # First, project separate predictions into separate representations
        representations = project_probabilities_vsa(predictions, embeddings, projection_mode)
        if (DEBUG):
            print_similarities(representations[i], embeddings, f"Predictor {i}")

        # Second, reduce separate representations into a single representation
        reduced_representations = reduce_vsa(representations, predictor_reduction.split("-")[1], axis=0)
        if (DEBUG):
            print_similarities(reduced_representations, embeddings, "Reduced")

        # Third, score programs based on the reduced single representation
        reduced_scored_programs = vsa_score(
            level=0,
            primitive=None,
            representation=reduced_representations,
            probability=1,
            embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            **kwargs,
        )

    # Perform reduce (mean, min, or max) then-project-then-score strategy
    elif ((predictor_reduction.lower().split("-")[1] == "project") and (predictor_reduction.lower().split("-")[2] == "score")):

        # First, reduce separate predictions into a single prediction
        reduced_predictions = reduce_probabilities(predictions, predictor_reduction.split("-")[0], axis=0)
        if (DEBUG):
            print_probabilities(reduced_predictions, "Reduced")

        # Second, project the single prediction into a single representation
        reduced_representations = project_probabilities_vsa(reduced_predictions, embeddings, projection_mode)
        if (DEBUG):
            print_similarities(reduced_representations, embeddings, "Reduced")

        # Third, score programs based on the reduced single representation
        reduced_scored_programs = vsa_score(
            level=0,
            primitive=None,
            representation=reduced_representations,
            probability=1,
            embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            **kwargs,
        )

    else:
        assert False, f"Invalid reduction over the predictor \"{predictor_reduction}\""

    # Fourth, reduce over constants, if needed
    if (predict_constants):
        reduced_constant_predictions = reduce_vsa(constant_representation, "Mean", axis=0)

    # Fifth, sort programs according to their single reduced score
    ranked_programs = sort_programs(reduced_scored_programs)
    if (DEBUG):
        print_programs(ranked_programs, "Reduced")

    if (predict_constants):
        return ranked_programs, reduced_constant_predictions
    else:
        return ranked_programs


def hybrid_inference_rank(
    demonstrations,
    prediction_nns,
    embeddings,
    projection_mode,    # "Matrix",
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    predictor_reduction,    # "Mean-Score",
    demonstration_reduction,    # "Mean",
    thumbs=None,
    predict_constants=False,
    **kwargs,
):
    if (predict_constants):
        primitive_predictions, constant_predictions = deepcoder_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
            predict_constants=True,
        )

        if (thumbs is not None):
            for p in thumbs:
                primitive_predictions[:, SORTED_PRIMITIVES.index(p)] = THUMBS_VALUE

        ranked_programs, reduced_constant_predictions = deepcoder_rank(
            primitive_predictions,
            embeddings=embeddings,
            projection_mode=projection_mode,
            predictor_reduction=predictor_reduction,
            predict_constants=True,
            constant_predictions=constant_predictions,
            **kwargs,
        )
        return ranked_programs, reduced_constant_predictions

    else:
        primitive_predictions = deepcoder_inference(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            demonstration_reduction=demonstration_reduction,
        )

        if (thumbs is not None):
            for p in thumbs:
                primitive_predictions[:, SORTED_PRIMITIVES.index(p)] = THUMBS_VALUE

        ranked_programs = hybrid_rank(
            primitive_predictions,
            embeddings=embeddings,
            projection_mode=projection_mode,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            **kwargs,
        )
        return ranked_programs
###############################################################################


# TASK SOLVING ALGORITHMS
###############################################################################
def try_programs_nofeedback(demonstrations, programs):
    for i, program in enumerate(programs):
        if (test_demonstrations(demonstrations, program)):
            if (DEBUG):
                print(f"Correct constant configuration found on attempt: {i + 1} / {len(programs)}")
            return program
    return None


def try_programs_feedback(
    demonstrations,
    programs,
    seed,
    demonstration_reduction,
    encoding_reduction,    # "Mean-Max"
    intermediate_reduction,    # "Rectify-Mean"
    intermediates=[],
    **kwargs,
):
    train_ins, train_outs = demonstrations

    best_program = None
    best_score = -float("inf")
    best_predictions = None
    for i, program in enumerate(programs):
        if (test_demonstrations(demonstrations, program)):
            if (DEBUG):
                print(f"Correct constant configuration found on attempt: {i + 1}/{len(programs)}")
            return True, {"program": program, "score": 1}

        predictions = [safe_execute_program(train_in, program) for train_in in train_ins]
        scores = np.array([
            score_similarity(
                baseline_ins,
                predictions,
                train_outs,
                seed=seed,
                encoding_reduction=encoding_reduction,
                demonstration_reduction=demonstration_reduction,
                **kwargs,
            ) for baseline_ins in ([train_ins] + intermediates)
        ])
        if (intermediate_reduction.lower() == "rectify-mean"):
            score = np.mean(scores) if (np.all(scores > 0)) else min(0, np.mean(scores))
        elif (intermediate_reduction.lower() == "mean"):
            score = np.mean(scores)
        else:
            assert False,  f"Invalid reduction over intermediates \"{intermediate_reduction}\""

        if (score > best_score):
            best_program = program
            best_score = score
            best_predictions = predictions

    return False, {"program": best_program, "score": best_score, "predictions": best_predictions}


def solve_task_by_enumeration(demonstrations, enumerated_programs):
    solution_programs = []
    for i, program in tqdm(enumerate(enumerated_programs)):
        if (DEBUG):
            print(f"Trying: {program} [{program.get_score():.3f}]")

        result = try_programs_nofeedback(demonstrations, assign_constants_to_program(program))
        if (result is not None):
            solution_programs.append((i + 1, result))
            print(f"Program found: `{result}` correctly solved all demonstrations ({i + 1}th attempt)")

        if (len(solution_programs) >= N_ATTEMPTS):
            break
    return solution_programs


def solve_task_by_evolution(
    demonstrations,
    predictions,
    mode,
    seed,
    predictor_reduction,
    update_weight,
    score_mode,    # Options: "Softplus", "Softmax"; Default: Unfiltered dot product
    subtract_mode,    # Options: "Scaled", "No Subtract"; Default: Subtract full representation
    beam_size,    # N_PRIMITIVES
    max_depth,    # MAX_DEPTH
    embeddings=None,
    **kwargs,
):

    attempted_programs = set()
    solution_programs = []
    for _ in range(100):
        if (mode.lower() == "deepcoder"):
            ranked_programs = deepcoder_rank(
                primitive_predictions=predictions,
                predictor_reduction=predictor_reduction,
                **kwargs,
            )

        elif (mode.lower() == "direct"):
            ranked_programs = vsa_rank(
                primitive_predictions=predictions,
                predictor_reduction=predictor_reduction,
                embeddings=embeddings,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                **kwargs,
            )

        elif (mode.lower() == "hybrid-deepcoder"):
            ranked_programs = hybrid_rank(
                primitive_predictions=predictions,
                predictor_reduction=predictor_reduction,
                embeddings=embeddings,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                **kwargs,
            )

        elif (mode.lower() == "hybrid-vsa"):
            ranked_programs = vsa_score(
                level=0,
                primitive=None,
                representation=predictions,
                probability=1,
                embeddings={p: embeddings[i] for i, p in enumerate(SORTED_PRIMITIVES)},
                **kwargs,
            )

        else:
            assert False, f"Invalid evolution mode \"{mode}\""

        ranked_programs = [program for program in ranked_programs if (program.get_program() not in attempted_programs)]
        program = ranked_programs[0]
        attempted_programs.add(program.get_program())

        if (DEBUG):
            print(f"Trying: {program} [{program.get_score():.3f}]")

        success, result = try_programs_feedback(demonstrations, assign_constants_to_program(program), seed=seed, **kwargs)

        if (success):
            solution_programs.append((len(attempted_programs), result["program"]))
            print(f"Program found: \"{result['program']}\" correctly solved all demonstrations ({len(attempted_programs)}th attempt)")

        if (len(solution_programs) >= N_ATTEMPTS):
            break

        if (mode.lower() == "deepcoder"):
            predictions += update_weight * result['score'] * np.array([(p in program.get_primitives()) for p in SORTED_PRIMITIVES])
            predictions = np.clip(predictions, 0, 1)
        
        elif (mode.lower() == "direct"):
            predictions += update_weight * result['score'] * program.compute_representation(embeddings)

        elif (mode.lower() == "hybrid-deepcoder"):
            predictions += update_weight * result['score'] * np.array([(p in program.get_primitives()) for p in SORTED_PRIMITIVES])
            predictions = np.clip(predictions, 0, 1)

        elif (mode.lower() == "hybrid-vsa"):
            predictions += update_weight * result['score'] * program.compute_representation(embeddings)

        else:
            assert False, f"Invalid evolution mode \"{mode}\""

    return solution_programs
###############################################################################


# BEST REASONER
###############################################################################
THUMBS_VALUE = 1
def get_similarity(
    input_grids,
    predicted_grids,
    output_grids,
    grid_encoding_function,
    formula,    # "Default"
    demonstration_reduction,    # "Mean",
    seed,
    **kwargs,    # Superfluous
):
    scores = np.zeros((len(input_grids)))
    for i, (input_grid, predicted_grid, output_grid) in enumerate(zip(input_grids, predicted_grids, output_grids)):
        input_grid_representation = grid_encoding_function(seed, input_grid)
        predicted_grid_representation = grid_encoding_function(seed, predicted_grid)
        output_grid_representation = grid_encoding_function(seed, output_grid)

        if (formula.lower() == "default"):
            scores[i] = (
                cosine_similarity(predicted_grid_representation, output_grid_representation) / 2 - \
                cosine_similarity(predicted_grid_representation, input_grid_representation) / 4 - \
                cosine_similarity(output_grid_representation, input_grid_representation) / 4
            )
        elif (formula.lower() == "output"):
            scores[i] = (
                cosine_similarity(predicted_grid_representation, output_grid_representation) / 2 - \
                cosine_similarity(output_grid_representation, input_grid_representation) / 2
            )
        elif (formula.lower() == "predicted"):
            scores[i] = (
                cosine_similarity(predicted_grid_representation, output_grid_representation) / 2 - \
                cosine_similarity(predicted_grid_representation, input_grid_representation) / 2
            )
        elif (formula.lower() == "unbind"):
            scores[i] = cosine_similarity(
                sspspace.bind(predicted_grid_representation, sspspace.invert(input_grid_representation)),
                sspspace.bind(output_grid_representation, sspspace.invert(input_grid_representation)),
            )
        else:
            assert False, f"Invalid formula \"{formula}\""

    score = reduce_scores(scores, demonstration_reduction)

    return score


def score_similarity(
    input_grids,
    predicted_grids,
    output_grids,
    encoding_reduction,    # "Mean",
    seed,
    encode_fcns,
    demonstration_reduction,
    formula,
    **kwargs,    # Superfluous
):
    # Check that the program did not throw an error
    if (any([(g.size == 0) for g in predicted_grids])):
        return -1

    scores = [
        get_similarity(
            input_grids=input_grids,
            predicted_grids=predicted_grids,
            output_grids=output_grids,
            grid_encoding_function=grid_encoding_function,
            seed=seed,
            demonstration_reduction=demonstration_reduction,
            formula=formula,
            **kwargs,
        ) for grid_encoding_function in encode_fcns
    ]

    score = reduce_scores(scores, encoding_reduction)

    return score


MAXIMUM_CONSIDERED_PROGRAMS = 100
MAXIMUM_ACTIVE_SET_SIZE = 25
SELECTION_TEMPERATURE = 2
def active_program_set_selector(
    programs,
    embeddings,
    temperature=SELECTION_TEMPERATURE,
):
    included_indices = [0]
    for i in range(1, min(MAXIMUM_CONSIDERED_PROGRAMS, len(programs))):
        ranking_score = np.exp(temperature * (1 - np.mean([programs[j].get_score() for j in included_indices]) / programs[i].get_score()))
        similarity_score = np.mean([np.dot(programs[j].compute_representation(embeddings, normalized=True), programs[i].compute_representation(embeddings, normalized=True)) for j in included_indices])
        if (ranking_score + similarity_score > 1):
            included_indices.append(i)
    included_indices = included_indices[:MAXIMUM_ACTIVE_SET_SIZE]
    active_program_set = [program for i, program in enumerate(programs) if (i in included_indices)]
    return active_program_set


def generate_active_program_set(
    demonstrations,
    prediction_nns,
    embeddings,
    partial_program_baselines,
    attempted_programs,
    fix_deep,
    projection_mode,    #"Matrix",
    score_mode,    # "Softplus",
    subtract_mode,    # "Conditioned",
    beam_size,    # N_PRIMITIVES,
    max_depth,    # 2,
    predictor_reduction,    # "Mean-Score",
    demonstration_reduction,    # "Mean",
    ranker,
    **kwargs,
):
    _, train_outs = demonstrations

    while (partial_program_baselines):
        if (ranker.lower() == "hybrid"):
            programs = hybrid_inference_rank(
                (partial_program_baselines[-1]["predictions"], train_outs),
                prediction_nns=prediction_nns,
                embeddings=embeddings,
                projection_mode=projection_mode,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                predictor_reduction=predictor_reduction,
                demonstration_reduction=demonstration_reduction,
                thumbs=partial_program_baselines[-1]["program"].get_primitives(),
                **kwargs,
            )
        elif (ranker.lower() == "deepcoder"):
            programs = deepcoder_inference_rank(
                (partial_program_baselines[-1]["predictions"], train_outs),
                prediction_nns=prediction_nns,
                embeddings=embeddings,
                projection_mode=projection_mode,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                predictor_reduction=predictor_reduction,
                demonstration_reduction=demonstration_reduction,
                thumbs=partial_program_baselines[-1]["program"].get_primitives(),
                **kwargs,
            )
        else:
            assert False

        programs = [p for p in programs if (p.get_program() not in attempted_programs)]
        if (fix_deep):
            programs = [p for p in programs if (all([(q in p.get_primitives()) for q in partial_program_baselines[-1]["program"].get_primitives()]))]
        ranked_programs = sorted(programs, key=(lambda x: -x.get_score()))

        # Want to ensure either (a) new programs are highly likely or (b) the improvement was marked enough
        ANOTHER_THRESH = 0.1
        if (ranked_programs) and ((ranked_programs[0].get_score() > ANOTHER_THRESH) or (partial_program_baselines[-1]["score"] > 0.1)):
            active_program_set = active_program_set_selector(
                programs,
                embeddings=embeddings,
            )
            return active_program_set

        partial_program_baselines.pop(-1)

    if (ranker.lower() == "hybrid"):
        programs = hybrid_inference_rank(
            demonstrations,
            prediction_nns=prediction_nns,
            embeddings=embeddings,
            projection_mode=projection_mode,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            demonstration_reduction=demonstration_reduction,
            thumbs=None,
            **kwargs,
        )
    elif (ranker.lower() == "deepcoder"):
        programs = deepcoder_inference_rank(
            demonstrations,
            prediction_nns=prediction_nns,
            embeddings=embeddings,
            projection_mode=projection_mode,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            demonstration_reduction=demonstration_reduction,
            thumbs=None,
            **kwargs,
        )
    else:
        assert False
    programs = [p for p in programs if (p.get_program() not in attempted_programs)]
    ranked_programs = sort_programs(programs)
    active_program_set = active_program_set_selector(programs, embeddings)
    return active_program_set


MAXIMUM_THINKING_STEPS = 10
MAXIMUM_ENUMERATION_STEPS = len(NAIVE_PROGRAMS)
def solve_task_by_GOAT(
    demonstrations,
    prediction_nns,
    embeddings,
    ranker,
    seed,
    score_mode,    # "Softplus",
    subtract_mode,    # "Conditioned",
    beam_size,    # N_PRIMITIVES,
    max_depth,    # 2,
    predictor_reduction,    # "Mean-Score",
    demonstration_reduction,    # "Mean",
    projection_mode,    # "Matrix",
    **kwargs,
):
    # print(solve_task_by_GOAT, kwargs)
    partial_program_baselines = []
    attempted_programs = set()
    solution_programs = []

    for n_thinking_step in range(MAXIMUM_THINKING_STEPS):
        active_program_set = generate_active_program_set(
            demonstrations=demonstrations,
            prediction_nns=prediction_nns,
            embeddings=embeddings,
            partial_program_baselines=partial_program_baselines,
            attempted_programs=attempted_programs,
            projection_mode=projection_mode,
            score_mode=score_mode,
            subtract_mode=subtract_mode,
            beam_size=beam_size,
            max_depth=max_depth,
            predictor_reduction=predictor_reduction,
            demonstration_reduction=demonstration_reduction,
            ranker=ranker,
            **kwargs,
        )
        # print(f"Active Set {n_thinking_step} (Based on \"{partial_program_baselines[-1]['program'] if (partial_program_baselines) else None}\"):",
        #     *[f"`{program}` [{program.get_score():.3f}]" for program in active_program_set], sep="\n\t")

        results = []
        for program in active_program_set:
            attempted_programs.add(program.get_program())

            success, result = try_programs_feedback(demonstrations, assign_constants_to_program(program), demonstration_reduction=demonstration_reduction, seed=seed, intermediates=[baseline["predictions"] for baseline in partial_program_baselines], **kwargs)
            results.append(result)

            if (success):
                solution_programs.append((len(attempted_programs), result["program"]))
                print(f"Program found: \"{result['program']}\" correctly solved all demonstrations ({len(attempted_programs)}th attempt)")

                if (len(solution_programs) >= N_ATTEMPTS):
                    break
        if (len(solution_programs) >= N_ATTEMPTS):
            break

        best_program = sorted(results, key=(lambda x: (-x["score"], -x["program"].get_score())))[0]
        IMPROVEMENT_THRESHOLD = 0
        if (best_program["score"] > IMPROVEMENT_THRESHOLD):
            # print(f"`{best_program['program']}` was an improvement with a score of {best_program['score']:.3f}")
            partial_program_baselines.append(best_program)

        else:
            # print(f"None were an improvement (top score was {best_program['score']:.3f})")
            pass

        # # Optional: Evolution-guiding the search
        # prediction += update_weight * np.sum([(np.sum(r[2]) * r[1].compute_representation(embeddings)) for r in results], axis=0)

    # Optional: Use naive enumeration to try remaining programs
    if (not solution_programs):
        if (ranker.lower() == "hybrid"):
            programs = hybrid_inference_rank(
                demonstrations,
                prediction_nns=prediction_nns,
                embeddings=embeddings,
                projection_mode=projection_mode,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                predictor_reduction=predictor_reduction,
                demonstration_reduction=demonstration_reduction,
                thumbs=None,
                **kwargs,
            )
        elif (ranker.lower() == "deepcoder"):
            programs = deepcoder_inference_rank(
                demonstrations,
                prediction_nns=prediction_nns,
                embeddings=embeddings,
                projection_mode=projection_mode,
                score_mode=score_mode,
                subtract_mode=subtract_mode,
                beam_size=beam_size,
                max_depth=max_depth,
                predictor_reduction=predictor_reduction,
                demonstration_reduction=demonstration_reduction,
                thumbs=None,
                **kwargs,
            )
        else:
            assert False
        programs = [p for p in programs if (p.get_program() not in attempted_programs)]
        ranked_programs = sort_programs(programs)
        solution_programs = solve_task_by_enumeration(demonstrations, ranked_programs[:MAXIMUM_ENUMERATION_STEPS])
        solution_programs = [(len(attempted_programs) + attempt_number, p) for attempt_number, p in solution_programs]

    return solution_programs
###############################################################################


def main():
    return


if __name__ == "__main__":
    main()
