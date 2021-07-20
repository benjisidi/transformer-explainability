# Imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.compute_gradients import (
    get_all_layer_gradients,
    get_all_layer_integrated_gradients,
)
from utils.compare_gradients import get_cos_similarities, get_n_best_matches
from pprint import pprint

# Memory Profiling
from pympler import asizeof

# Timings
import sys

sys.path.append("/home/benji/repos/tachymeter/chronograph")
from chronograph import Chronograph

cg = Chronograph()


from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset("glue", "sst2")
    #  Get test samples
    test_sample = dataset["validation"][0]["sentence"]
    test_label = dataset["validation"][0]["label"]
    train_samples = dataset["train"][:200]["sentence"]
    train_labels = dataset["train"][:200]["label"]
    # Define Model
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model.eval()
    # Test Layerwise Gradients
    cg.start("Grads")
    grads = get_all_layer_gradients(train_samples, train_labels, model, tokenizer)
    # Test LIG
    cg.split("ligs")
    ligs = get_all_layer_integrated_gradients(
        [test_sample], [test_label], model, tokenizer
    )
    cg.split("Similarities")
    simils = get_cos_similarities(ligs, grads)
    best_examples = get_n_best_matches(simils, train_samples)
    cg.stop()

    print(test_sample)
    pprint(best_examples)
    print(cg.report())
