import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot Training and Evaluation")
    # Dataset settings
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # General training settings
    parser.add_argument(
        "--pretrainer_epochs",
        type=int,
        default=10000,
        help="Number of pretraining epochs",
    )
    parser.add_argument(
        "--trainer_epochs",
        type=int,
        default=10000,
        help="Number of classification training epochs",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (steps)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizers"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for all loaders"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience"
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="Number of support examples per class"
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=100,
        help="Evaluate model for N steps during testing",
    )

    # Model settings
    parser.add_argument(
        "--masked_ratio", type=float, default=0.2, help="Masking ratio for pretraining"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension of backbone network",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="Embedding dimension in predictor network",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Contrastive temperature"
    )
    parser.add_argument(
        "--ensemble",
        choices=["True", "False"],
        default="False",
        help="Use ensemble of models",
    )
    parser.add_argument(
        "--classifier_model",
        choices=["probe", "proto", "nn"],
        default="probe",
        help="Classifier model for evaluation",
    )
    parser.add_argument(
        "--fill_mode",
        choices=["zero", "marginal"],
        default="zero",
        help="Use marginal distribution or zero to fill masked values",
    )
    parser.add_argument(
        "--metric_clf_mode",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Metric for nn and proto classification",
    )

    args = parser.parse_args()
    args.ensemble = args.ensemble == "True"
    return args
