# Isaac Joffe, 2026


from utils import *
import data_generator


# DEEPCODER-STYLE PREDICTION NN
###############################################################################
class DeepCoderNN(torch.nn.Module):
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=128,
        output_dim=32,
        n_hidden_layers=1,
        hidden_layer_sizes="linear",
        n_epochs=1000,
        early_stopping=True,
        stopping_window=10,
        batch_size=256,
        learning_rate=0.0001,
        regularization_factor=1,
        dropout_factor=0.5,
        label_smoothing_factor=0,
        grid_encoding_function=(lambda x: x),
        seed=0,
    ):
        super().__init__()

        # Architecture hyperparameters
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__n_hidden_layers = n_hidden_layers
        self.__hidden_layer_sizes = hidden_layer_sizes

        # Training hyperparameters
        self.__n_epochs = n_epochs
        self.__early_stopping = early_stopping
        self.__stopping_window = stopping_window
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate

        # Regularization hyperparameters
        self.__regularization_factor = regularization_factor
        self.__dropout_factor = dropout_factor
        self.__label_smoothing_factor = label_smoothing_factor

        # Prediction hyperparameters
        self.__grid_encoding_function = grid_encoding_function
        self.__seed = seed

        # Determine the sizes of the hidden layers
        if (self.__hidden_layer_sizes.lower() == "linear"):
            layer_dims = np.linspace(self.__input_dim, self.__hidden_dim, self.__n_hidden_layers + 1, dtype=int)
        elif (self.__hidden_layer_sizes.lower() == "log"):
            layer_dims = np.logspace(np.log2(self.__input_dim), np.log2(self.__hidden_dim), self.__n_hidden_layers + 1, base=2, dtype=int)
        else:
            assert False, f"Invalid mode {self.__hidden_layer_sizes} for hidden layer dimensions"

        # Hidden layers
        layers = []
        for i in range(self.__n_hidden_layers):
            layers.append(
                torch.nn.Linear(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                ),
            )
            layers.append(
                torch.nn.ReLU(),
            )
            layers.append(
                torch.nn.Dropout(
                    p=self.__dropout_factor,
                ),
            )

        # Output layer
        layers.append(
            torch.nn.Linear(
                in_features=layer_dims[-1],
                out_features=self.__output_dim,
            ),
        )
        layers.append(
            torch.nn.Sigmoid(),
        )

        # Construct NN
        self.nn = torch.nn.Sequential(
            *layers,
        )

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.__learning_rate)
        return

    def print_nn(self):
        print(f"{self.__class__.__name__} hyperparameters:")
        attributes = {name: value for name, value in vars(self).items() if (name.startswith("_DeepCoderNN__"))}
        print(*[f"\t{name.replace('_DeepCoderNN__', '')}: {value}" for name, value in attributes.items()], sep="\n")
        print(f"{self.__class__.__name__} architecture:")
        print("\t" + str(self.nn).replace("\n", "\n\t"))
        return

    def print_parameter_count(self):
        n_parameters = sum(p.numel() for p in self.parameters() if (p.requires_grad))
        print(f"{self.__class__.__name__} parameter count: {n_parameters:,}")
        return n_parameters

    def smooth_labels(self, x):
        return x.clamp(self.__label_smoothing_factor, 1 - self.__label_smoothing_factor)

    def forward(self, x):
        return self.nn(x)

    def predict(self, x1, x2):
        return self.nn(
            torch.concat(
                (
                    torch.Tensor(self.__grid_encoding_function(self.__seed, x1)),
                    torch.Tensor(self.__grid_encoding_function(self.__seed, x2)),
                ),
                axis=1,
            ),
        ).detach().numpy()

    def learn(self, train_data, val_data=None, visualize=True):
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.__batch_size,
            shuffle=True,
            pin_memory=(device == "cuda"),
            num_workers=0,
        )

        train_losses = []
        val_losses = []
        rolling_losses = []

        iteration = 0
        for epoch in tqdm(range(self.__n_epochs)):

            # Training loop
            self.train()
            for train_in_batch, train_out_batch in train_dataloader:
                iteration += 1

                train_loss = torch.nn.BCELoss(reduction="sum")(self(train_in_batch.to(device)), self.smooth_labels(train_out_batch.to(device)))
                train_losses.append((iteration, train_loss.detach().numpy() / len(train_in_batch)))
                train_loss += self.__regularization_factor * sum([torch.norm(self.nn[i].weight) for i in range(len(self.nn)) if ("weight" in self.nn[i].state_dict())])

                self.optimizer.zero_grad(set_to_none=True)
                train_loss.backward()
                self.optimizer.step()

            # Validation loop
            if (val_data is not None):
                self.eval()

                with torch.no_grad():
                    val_loss = torch.nn.BCELoss(reduction="sum")(self(val_data.tensors[0].to(device)), self.smooth_labels(val_data.tensors[1].to(device)))
                    val_losses.append((iteration, val_loss.detach().numpy() / len(val_data)))

                if (epoch > self.__stopping_window):
                    rolling_losses.append((iteration, sum([l[1] for l in val_losses[-self.__stopping_window:]]) / self.__stopping_window))
                    if (self.__early_stopping) and (len(rolling_losses) > self.__stopping_window) and (rolling_losses[-1][1] > rolling_losses[-self.__stopping_window][1]):
                        break

        if (visualize):
            plt.clf()
            plt.plot(*(np.array(train_losses).T), label="Training Loss")
            plt.plot(*(np.array(val_losses).T), label="Validation Loss")
            plt.plot(*(np.array(rolling_losses).T), label="Rolling Average Validation Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss Per Sample")
            plt.title("Loss Curve")
            plt.legend()
            plt.gcf().canvas.flush_events()
            plt.show()
            plt.close()

            learned_embeddings = self.nn[-2].weight.detach().numpy()
            normalized_learned_embeddings = (learned_embeddings.T / np.linalg.norm(learned_embeddings, axis=1)).T
            embedding_similarity_map = plt.pcolormesh(np.array(normalized_learned_embeddings @ normalized_learned_embeddings.T))
            plt.colorbar(embedding_similarity_map)
            plt.yticks(np.arange(len(SORTED_PRIMITIVES))+0.5, [p.__name__ for p in SORTED_PRIMITIVES])
            plt.title("Similarity Map of Learned Embeddings")
            plt.gcf().canvas.flush_events()
            plt.show()
            plt.close()

            two_dimensional_embeddings = sklearn.decomposition.PCA(n_components=2).fit_transform(learned_embeddings)
            plt.clf()
            plt.scatter(*two_dimensional_embeddings.T)
            for i, p in enumerate(SORTED_PRIMITIVES):
                plt.text(*two_dimensional_embeddings[i], p.__name__)
            plt.title("Learned Embeddings of DSL Primitives in 2D")
            plt.gcf().canvas.flush_events()
            plt.show()
            plt.close()

            three_dimensional_embeddings = sklearn.decomposition.PCA(n_components=3).fit_transform(learned_embeddings)
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(*three_dimensional_embeddings.T)
            for i, p in enumerate(SORTED_PRIMITIVES):
                ax.text(*three_dimensional_embeddings[i], p.__name__)
            plt.title("Learned Embeddings of DSL Primitives in 3D")
            plt.gcf().canvas.flush_events()
            plt.show()
            plt.close()

            plt.clf()
            plt.plot(np.linalg.svd(learned_embeddings)[1], marker="o")
            plt.title("Singular Values")
            plt.gcf().canvas.flush_events()
            plt.show()
            plt.close()

        self.eval()
        return


def train_deepcoder_nn(
    train_dataset_name,
    val_dataset_name,
    input_dim=2048,
    hidden_dim=128,
    output_dim=32,
    n_hidden_layers=1,
    hidden_layer_sizes="linear",
    n_epochs=1000,
    early_stopping=True,
    stopping_window=10,
    batch_size=256,
    learning_rate=0.0001,
    regularization_factor=1,
    dropout_factor=0.5,
    label_smoothing_factor=0,
    grid_encoding_function=(lambda x: x),
    seed=0,
):
    rng = reseed(seed)

    train_data = torch.load(f"{DATA_PATH}{train_dataset_name}", map_location=device, weights_only=False)
    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])

    val_data = torch.load(f"{DATA_PATH}{val_dataset_name}", map_location=device, weights_only=False)
    val_dataset = torch.utils.data.TensorDataset(val_data[0], val_data[1])

    assert (train_dataset.tensors[0].shape[1] == input_dim), f"Training dataset and NN have an input size mismatch: {train_dataset.tensors[0].shape} != {input_dim}"
    assert (train_dataset.tensors[1].shape[1] == output_dim), f"Training dataset and NN have an output size mismatch: {train_dataset.tensors[1].shape} != {output_dim}"
    assert (val_dataset.tensors[0].shape[1] == input_dim), f"Validation dataset and NN have an input size mismatch: {val_dataset.tensors[0].shape} != {input_dim}"
    assert (val_dataset.tensors[1].shape[1] == output_dim), f"Validation dataset and NN have an output size mismatch: {val_dataset.tensors[1].shape} != {output_dim}"

    deepcoder_nn = DeepCoderNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_hidden_layers=n_hidden_layers,
        hidden_layer_sizes=hidden_layer_sizes,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        stopping_window=stopping_window,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization_factor=regularization_factor,
        dropout_factor=dropout_factor,
        label_smoothing_factor=label_smoothing_factor,
        grid_encoding_function=grid_encoding_function,
        seed=seed,
    ).to(device)
    deepcoder_nn.print_nn()
    deepcoder_nn.print_parameter_count()

    deepcoder_nn.learn(
        train_dataset,
        val_dataset,
        visualize=False,
    )

    return deepcoder_nn
###############################################################################


# EMBEDDINGS SETUP
###############################################################################
def generate_embeddings(
    embedding_dim=1024,
    source="Random",
    source_embeddings=None,
    ssp_latent_dim=3,
    ssp_length_scale=0.25,
    normalized=True,
    type_tags=False,
    type_tags_mode="Basic",
    seed=0,
):
    rng = reseed(seed)

    # First, construct from source (random, learned, or SSP)
    if (source.lower() == "random"):
        embeddings = np.random.randn(N_PRIMITIVES, embedding_dim)

    elif (source.lower() == "learned"):
        assert (source_embeddings is not None), "Must provide source embeddings for \"Learned\" mode"
        embeddings = source_embeddings

    elif (source.lower() == "ssp"):
        assert (source_embeddings is not None), "Must provide source embeddings for \"Learned\" mode"
        PRIMITIVE_SSP_SPACE = sspspace.RandomSSPSpace(
            domain_dim=ssp_latent_dim,
            ssp_dim=embedding_dim,
            length_scale=ssp_length_scale,
            kernel="gaussian",
            rng=rng,
        )
        if (source_embeddings.shape[1] != ssp_latent_dim):
            embeddings = sklearn.decomposition.PCA(n_components=ssp_latent_dim).fit_transform(source_embeddings)
        else:
            embeddings = source_embeddings
        embeddings = PRIMITIVE_SSP_SPACE.encode(embeddings)

    assert (embeddings.shape == (N_PRIMITIVES, embedding_dim)), "Embeddings are not the proper size"

    # Second, if desired, normalize the embeddings to unit length (assumption of search algorithm)
    if (normalized):
        embeddings = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T
        assert (np.allclose(np.linalg.norm(embeddings, axis=1), 1)), "Embeddings are not unit-length"

    # Third, if desired, bind representations with typetags
    if (type_tags):
        if (type_tags_mode.lower() == "basic"):
            TYPE_TAGS_SP_SPACE = sspspace.DiscreteSPSpace(
                keys=SORTED_TYPES,
                ssp_dim=embedding_dim,
            )
            TYPE_TAGS = {t: TYPE_TAGS_SP_SPACE.map[i] for i, t in enumerate(SORTED_TYPES)}
            for i in range(len(embeddings)):
                embeddings[i] = sspspace.bind(
                    np.atleast_2d(embeddings[i]),
                    np.atleast_2d(TYPE_TAGS[SORTED_PRIMITIVES[i].get_return_type()]),
                ).flatten()

        elif (type_tags_mode.lower() == "correlated"):
            keys = [t.__name__ for t in SORTED_TYPES] + [f"{t.__name__}-Common" for t in SORTED_TYPES]
            TYPE_TAGS_SP_SPACE = sspspace.DiscreteSPSpace(
                keys=keys,
                ssp_dim=embedding_dim,
            )
            common_vector = np.sum(TYPE_TAGS_SP_SPACE.map[len(TYPES):], axis=0)
            COMMON_WEIGHT = 1
            DIFFERENT_WEIGHT = 5
            TYPE_TAGS = {t: TYPE_TAGS_SP_SPACE.map[i] + (COMMON_WEIGHT + DIFFERENT_WEIGHT) * TYPE_TAGS_SP_SPACE.map[i + len(TYPES)] - COMMON_WEIGHT * common_vector for i, t in enumerate(SORTED_TYPES)}
            TYPE_TAGS = {t: (TYPE_TAGS[t] / np.linalg.norm(TYPE_TAGS[t])) for t in TYPE_TAGS}
            for i in range(len(embeddings)):
                embeddings[i] = sspspace.bind(
                    np.atleast_2d(embeddings[i]),
                    np.atleast_2d(TYPE_TAGS[SORTED_PRIMITIVES[i].get_return_type()]),
                ).flatten()

    return embeddings


def random_embeddings(
    embedding_dim,
    seed,
):
    # Random embeddings require no source
    # Random embeddings require no SSP hyperparameters
    # Random embeddings must be normalized
    # Random embeddings gain nothing from type tags
    return generate_embeddings(
        embedding_dim=embedding_dim,
        source="Random",
        source_embeddings=None,
        ssp_latent_dim=None,
        ssp_length_scale=None,
        normalized=True,
        type_tags=False,
        type_tags_mode=None,
        seed=seed,
    )


def learned_embeddings(
    source_embeddings,
    normalized,
    type_tags,
    type_tags_mode,
    seed,
):
    # Learned embeddings must have the same dimensions as source embedding
    # Learned embeddings require no SSP hyperparameters
    return generate_embeddings(
        embedding_dim=source_embeddings.shape[1],
        source="Learned",
        source_embeddings=source_embeddings,
        ssp_latent_dim=None,
        ssp_length_scale=None,
        normalized=normalized,
        type_tags=type_tags,
        type_tags_mode=type_tags_mode,
        seed=seed,
    )


def ssp_embeddings(
    embedding_dim,
    source_embeddings,
    ssp_latent_dim,
    ssp_length_scale,
    type_tags,
    type_tags_mode,
    seed,
):
    # SSP embeddings are necessarily normalized
    return generate_embeddings(
        embedding_dim=embedding_dim,
        source="SSP",
        source_embeddings=source_embeddings,
        ssp_latent_dim=ssp_latent_dim,
        ssp_length_scale=ssp_length_scale,
        normalized=True,
        type_tags=type_tags,
        type_tags_mode=type_tags_mode,
        seed=seed,
    )
###############################################################################


def main():
    DEEPCODER_HYPERPARAMS = {
        "input_dim": int(2 * GRID_REPRESENTATION_DIMENSIONS),
        "hidden_dim": 128,
        "output_dim": N_PRIMITIVES,
        "n_hidden_layers": 2,
        "hidden_layer_sizes": "log",
        "n_epochs": 1000,
        "early_stopping": True,
        "stopping_window": 10,
        "batch_size": 256,
        "learning_rate": 0.0001,
        "regularization_factor": 1,
        "dropout_factor": 0.5,
        "label_smoothing_factor": 0.01,
    }

    for seed in SEEDS:
        for scale in [1000, 10000]:

            with open(f"{DATA_PATH}Apr16_{scale}N_0S.json", "r") as f:
                raw_train_dataset = json.load(f)
            with open(f"{DATA_PATH}Apr16_{scale // 2}N_0S.json", "r") as f:
                raw_val_dataset = json.load(f)

            for mode in ["1", "2", "3", "All", "Naive"]:

                print(f"Seed: {seed}; Scale: {scale}; Mode: {mode}")

                train_dataset = torch.utils.data.TensorDataset(
                    torch.Tensor(data_generator.encode_grids([np.array(sample["input_grid"]) for sample in raw_train_dataset], [np.array(sample["output_grid"]) for sample in raw_train_dataset], lambda x: eval(f"encode_grid_{mode.lower()}")(seed, x))),
                    torch.Tensor(data_generator.encode_primitives([MyProgram(sample["program"], [eval(p) for p in sample["primitives"]]) for sample in raw_train_dataset])),
                )
                val_dataset = torch.utils.data.TensorDataset(
                    torch.Tensor(data_generator.encode_grids([np.array(sample["input_grid"]) for sample in raw_val_dataset], [np.array(sample["output_grid"]) for sample in raw_val_dataset], lambda x: eval(f"encode_grid_{mode.lower()}")(seed, x))),
                    torch.Tensor(data_generator.encode_primitives([MyProgram(sample["program"], [eval(p) for p in sample["primitives"]]) for sample in raw_val_dataset])),
                )

                reseed(seed)

                deepcoder_nn = DeepCoderNN(
                    **DEEPCODER_HYPERPARAMS,
                ).to(device)
                deepcoder_nn.print_nn()
                deepcoder_nn.print_parameter_count()

                deepcoder_nn.learn(
                    train_dataset,
                    val_dataset,
                    visualize=False,
                )
                file_path = f"{MODELS_PATH}New_{mode}R_{scale}N_{seed}S.pt"
                torch.save(deepcoder_nn.state_dict(), file_path)
                print(file_path)

    return


if __name__ == "__main__":
    main()
