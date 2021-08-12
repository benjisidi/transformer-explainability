# %%
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

# %%
with open("../dense_gradients.pkl", "rb") as f:
    data = pickle.load(f)

data.shape

# %%
normalized_data = torch.nn.functional.normalize(data, dim=1)
# %%
test_example = normalized_data[:, 0, :]
# %%
test_example
# %%
sb.histplot(test_example.flatten().numpy())
# %%
sb.histplot(data[:, 0, :].flatten().numpy())

# %%
test_examples = normalized_data[:, :200, :]
sb.histplot(test_examples.flatten().numpy())
stats.describe(test_examples.flatten().numpy())

# %%
test_examples_raw = data[:, :200, :]

sb.histplot(test_examples_raw.flatten().numpy())
stats.describe(test_examples_raw.flatten().numpy())
# %%
# %%
means = torch.mean(normalized_data, dim=1)
vars = torch.var(normalized_data, dim=1)

# %%
