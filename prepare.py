import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

# 1. Define features
NUMERIC = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]

CATEGORICAL = ["station_type"]
FEATURES = NUMERIC + CATEGORICAL


# 2. Create a custom PyTorch Dataset
class BikeDataset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 3. Split and prepare the data
def prepare_data(seed=42):
    df = make_rentals()

    # Split 60% training and 40% remaining
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
    )

    # Divide the remaining rows equally
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        random_state=seed,
    )

    # Scale numbers and encode station types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
        ]
    )

    # CRITICAL: fit only on training rows
    preprocessor.fit(train_df[FEATURES])

    def to_dataset(part):
        x = preprocessor.transform(part[FEATURES])
        y = part[TARGET].to_numpy()
        return BikeDataset(x, y)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train": to_dataset(train_df),
        "val": to_dataset(val_df),
        "test": to_dataset(test_df),
        "preprocessor": preprocessor,
    }


# 4. Create batches for the network
def make_loaders(data, seed=42, batch_size=64):
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        data["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        data["val"],
        batch_size=256,
        shuffle=False,
    )

    test_loader = DataLoader(
        data["test"],
        batch_size=256,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# 5. Check the results
if __name__ == "__main__":
    data = prepare_data()

    train_loader, val_loader, test_loader = make_loaders(data)

    print("Train:", len(data["train"]))
    print("Validation:", len(data["val"]))
    print("Test:", len(data["test"]))

    # Confirm no row belongs to multiple sets
    train_ids = set(data["train_df"].index)
    val_ids = set(data["val_df"].index)
    test_ids = set(data["test_df"].index)

    no_overlap = (
        train_ids.isdisjoint(val_ids)
        and train_ids.isdisjoint(test_ids)
        and val_ids.isdisjoint(test_ids)
    )
    print("No overlapping rows:", no_overlap)

    # Look at one batch
    xb, yb = next(iter(train_loader))

    print("Feature batch shape:", xb.shape)
    print("Target batch shape:", yb.shape)
    print("Feature dtype:", xb.dtype)
    print("Target dtype:", yb.dtype)

    # Numeric features should be centered on training rows
    n_numeric = len(NUMERIC)
    train_mean = data["train"].x[:, :n_numeric].mean(dim=0)
    val_mean = data["val"].x[:, :n_numeric].mean(dim=0)

    print("Training numeric means:", train_mean.round(decimals=4))
    print("Validation numeric means:", val_mean.round(decimals=4))
