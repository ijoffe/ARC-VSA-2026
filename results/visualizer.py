# Isaac Joffe, 2026


import numpy as np
import matplotlib.pyplot as plt


N_TASKS = 53
N_PROGRAMS = 29700


def plot_task_solving_efficiency(
    data,
    name,
    size=0.49,
):
    print("Scores (Average Area Under the Curve):")
    scores = {}

    IMAGE_SIZE = 6 * size
    plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE * 2 / (1 + np.sqrt(5))))

    for label in data:
        logx = np.logspace(0, np.log10(N_PROGRAMS), 100)
        precise_logx = np.logspace(0, np.log10(N_PROGRAMS), 1000)

        cumulative_results = np.array([np.sum(np.logical_and(data[label] > 0, data[label] < x), axis=0) for x in logx])
        precise_cumulative_results = np.array([np.sum(np.logical_and(data[label] > 0, data[label] < x), axis=0) for x in precise_logx])

        mean_cumulative_results = np.mean(cumulative_results, axis=1)
        mean_precise_cumulative_results = np.mean(precise_cumulative_results, axis=1)
        min_cumulative_results = np.min(cumulative_results, axis=1)
        max_cumulative_results = np.max(cumulative_results, axis=1)

        scores[label] = np.trapezoid(mean_precise_cumulative_results.T, np.log10(precise_logx)) / np.trapezoid(np.full((len(precise_logx)), N_TASKS), np.log10(precise_logx))
        print(f"\t{label.rjust(max([len(l) for l in data.keys()]))}: {scores[label]:.3f}")

        if (label == "Baseline"):
            plt.plot(logx, mean_cumulative_results, label=f"{label} ({scores[label]:.3f})", color="#AAAAAA")
        else:
            plt.plot(logx, mean_cumulative_results, label=f"{label}({scores[label]:.3f})")
            plt.fill_between(logx, min_cumulative_results, max_cumulative_results, alpha=0.5)

    print()

    plt.xscale("log")
    plt.gca().set_xticks([1, 10, 100, 1000, 10000])
    plt.gca().set_xticks([
        2, 3, 4, 5, 6, 7, 8, 9,
        20, 30, 40, 50, 60, 70, 80, 90,
        200, 300, 400, 500, 600, 700, 800, 900,
        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
        20000,
    ], minor=True)
    plt.xlabel("Programs Proposed")

    plt.ylabel("Tasks Solved")

    plt.legend(loc="lower right", edgecolor="k", framealpha=1)
    plt.title("Task-Solving Efficiency")
    plt.savefig(f"../paper/media/{name}.pdf")
    plt.show()
    plt.close()

    return scores


data_1k = np.load("1K_Data.npz")
data_10k = np.load("10K_Data.npz")


# Result 0: Best/naive/brute-force comparisons
best = data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"]
naive = data_10k["DeepCoder-Style Enumeration - Naive"]
baseline = data_10k["Naive Baseline"]

print(f"Best @10: {np.sum(best <= 10, axis=0)} ({int(np.median(np.sum(best <= 10, axis=0)))})")
print(f"Naive @10: {np.sum(naive <= 10, axis=0)} ({int(np.median(np.sum(naive <= 10, axis=0)))})")
print()
print(f"Best @100: {np.sum(best <= 100, axis=0)} ({int(np.median(np.sum(best <= 100, axis=0)))})")
print(f"Naive @100: {np.sum(naive <= 100, axis=0)} ({int(np.median(np.sum(naive <= 100, axis=0)))})")
print()
print(f"Best/Naive @Median: {int(np.round(np.median(np.median(naive, axis=1)) / np.median(np.median(best, axis=1))))}")
print(f"Best/Brute @Median: {int(np.round(np.median(np.median(baseline, axis=1)) / np.median(np.median(best, axis=1))))}")
print()
print(f"Best/Naive @Mean: {int(np.round(np.mean(np.median(naive, axis=1)) / np.mean(np.median(best, axis=1))))}")
print(f"Best/Brute @Mean: {int(np.round(np.mean(np.median(baseline, axis=1)) / np.mean(np.median(best, axis=1))))}")
print()


# Result 1: Ensembling Helps
plot_task_solving_efficiency(
    {
        "Best Ens.~Enum.~": data_1k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Best Comb.~Enum.~": data_1k["Hybrid-VSA Enumeration - All - Random"],
        "Best Naive Enum.~": data_1k["Hybrid-VSA Enumeration - Naive - Random"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result1_enumeration_1k",
)
plot_task_solving_efficiency(
    {
        "Best Ens.~Enum.~": data_10k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Best Comb.~Enum.~": data_10k["Hybrid-VSA Enumeration - All - Random"],
        "Best Naive Enum.~": data_10k["Hybrid-VSA Enumeration - Naive - Random"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result1_enumeration_10k",
)
plot_task_solving_efficiency(
    {
        "Best Ens.~Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Comb.~Evo.~": data_1k["Hybrid GOAT Evolution - All - Random"],
        "Best Naive Evo.~": data_1k["Hybrid GOAT Evolution - Naive - Random"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result1_evolution_1k",
)
plot_task_solving_efficiency(
    {
        "Best Ens.~Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Comb.~Evo.~": data_10k["Hybrid GOAT Evolution - All - Random"],
        "Best Naive Evo.~": data_10k["Hybrid GOAT Evolution - Naive - Engineered SSP"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result1_evolution_10k",
)


# Result 2: VSAs do not hurt
plot_task_solving_efficiency(
    {
        "Best VSA Enum.~": data_1k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Best Naive Enum.~": data_1k["DeepCoder-Style Enumeration Ensemble - Max-Score"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result2_enumeration_1k",
)
plot_task_solving_efficiency(
    {
        "Best VSA Enum.~": data_10k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Best Naive Enum.~": data_10k["DeepCoder-Style Enumeration Ensemble - Max-Score"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result2_enumeration_10k",
)
plot_task_solving_efficiency(
    {
        "Best VSA Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Naive Evo.~": data_1k["DeepCoder-Style GOAT Evolution Ensemble - Mean-Score"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result2_evolution_1k",
)
plot_task_solving_efficiency(
    {
        "Best VSA Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Naive Evo.~": data_10k["DeepCoder-Style GOAT Evolution Ensemble - Score-Mean"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result2_evolution_10k",
)


# Result 3: Evolution helps significantly
plot_task_solving_efficiency(
    {
        "Best Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Enum.~": data_1k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Naive~": data_1k["DeepCoder-Style Enumeration - Naive"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result3_engineered_1k",
)
plot_task_solving_efficiency(
    {
        "Best Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Enum.~": data_10k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Naive~": data_10k["DeepCoder-Style Enumeration - Naive"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result3_engineered_10k",
)
plot_task_solving_efficiency({
        "Best Eng.~Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Enum.~": data_1k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Naive~": data_1k["DeepCoder-Style Enumeration - Naive"],
        "Best Random Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Random"],
        "Best Learned Evo.~": data_1k["Hybrid GOAT Evolution Ensemble - Project-Score-Max - Typetagged Learned SSP"],
        "Baseline": data_1k["Naive Baseline"],
    },
    "result3_learned_1k",
)
plot_task_solving_efficiency({
        "Best Eng.~Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Engineered SSP"],
        "Best Enum.~": data_10k["Hybrid-VSA Enumeration Ensemble - Max-Project-Score - Random"],
        "Naive~": data_10k["DeepCoder-Style Enumeration - Naive"],
        "Best Random Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Mean - Random"],
        "Best Learned Evo.~": data_10k["Hybrid GOAT Evolution Ensemble - Project-Score-Max - Typetagged Learned SSP"],
        "Baseline": data_10k["Naive Baseline"],
    },
    "result3_learned_10k",
)
