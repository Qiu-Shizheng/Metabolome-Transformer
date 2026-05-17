import argparse

from .finetune import predict_binary_classifier, train_binary_classifier
from .imputation import impute_missing_values
from .io_utils import read_table, write_table
from .pretrained import MetabolomeTransformerPipeline


def add_runtime_args(parser):
    parser.add_argument("--device", default="auto", help="Device used for inference or training.")


def add_finetune_model_args(parser):
    parser.add_argument("--checkpoint", default=None, help="Path to pretrained checkpoint.")
    parser.add_argument("--hparams", default=None, help="Path to model hyperparameter JSON.")
    parser.add_argument("--embeddings", default=None, help="Path to metabolite semantic embeddings.")
    parser.add_argument("--metabolite-columns", default=None, help="Path to metabolite column list.")
    add_runtime_args(parser)


def build_pipeline(args):
    kwargs = {"device": args.device}
    if hasattr(args, "checkpoint") and args.checkpoint:
        kwargs["ckpt_path"] = args.checkpoint
    if hasattr(args, "hparams") and args.hparams:
        kwargs["hparams_path"] = args.hparams
    if hasattr(args, "embeddings") and args.embeddings:
        kwargs["semantic_embedding_path"] = args.embeddings
    if hasattr(args, "metabolite_columns") and args.metabolite_columns:
        kwargs["metabolite_columns_path"] = args.metabolite_columns
    return MetabolomeTransformerPipeline(**kwargs)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="metabolome-transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    formatter = argparse.ArgumentDefaultsHelpFormatter

    p_imp = subparsers.add_parser(
        "impute",
        help="Impute missing metabolomics values",
        formatter_class=formatter,
    )
    p_imp.add_argument("--input", required=True, help="Input CSV/TSV table.")
    p_imp.add_argument("--output", required=True, help="Output table path.")
    p_imp.add_argument("--id-column", default=None, help="Sample ID column. Uses the first column if omitted.")
    p_imp.add_argument("--batch-size", type=int, default=12, help="Inference batch size.")
    add_runtime_args(p_imp)

    p_train = subparsers.add_parser(
        "train-binary",
        help="Fine-tune a current or future binary disease classifier",
        formatter_class=formatter,
    )
    p_train.add_argument("--input", required=True, help="Training CSV/TSV table.")
    p_train.add_argument("--output-dir", required=True, help="Directory for the fine-tuned classifier.")
    p_train.add_argument("--id-column", default=None, help="Sample ID column. Uses the first column if omitted.")
    p_train.add_argument("--label-column", required=True, help="Binary label column.")
    p_train.add_argument("--task", choices=["current", "future"], default="current", help="Prediction task type.")
    p_train.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    p_train.add_argument("--batch-size", type=int, default=12, help="Training batch size.")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p_train.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    p_train.add_argument("--val-fraction", type=float, default=0.2, help="Validation split fraction.")
    p_train.add_argument("--freeze-backbone", action="store_true", help="Train only the binary head.")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed for the validation split.")
    add_finetune_model_args(p_train)

    p_pred = subparsers.add_parser(
        "predict-binary",
        help="Predict with a fine-tuned binary classifier",
        formatter_class=formatter,
    )
    p_pred.add_argument("--input", required=True, help="Prediction CSV/TSV table.")
    p_pred.add_argument("--classifier", required=True, help="Fine-tuned classifier checkpoint.")
    p_pred.add_argument("--output", required=True, help="Output prediction table.")
    p_pred.add_argument("--id-column", default=None, help="Sample ID column. Uses the first column if omitted.")
    p_pred.add_argument("--batch-size", type=int, default=12, help="Inference batch size.")
    add_finetune_model_args(p_pred)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    pipeline = build_pipeline(args)

    if args.command == "impute":
        df = read_table(args.input)
        aligned = pipeline.align_input_dataframe(df, id_column=args.id_column)
        imputed = impute_missing_values(pipeline, aligned, id_column=args.id_column, batch_size=args.batch_size)
        write_table(imputed, args.output)

    elif args.command == "train-binary":
        df = read_table(args.input)
        aligned = pipeline.align_input_dataframe(df, id_column=args.id_column)
        aligned[args.label_column] = df[args.label_column].values
        train_binary_classifier(
            pipeline,
            aligned,
            label_column=args.label_column,
            output_dir=args.output_dir,
            task=args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            freeze_backbone=args.freeze_backbone,
            seed=args.seed,
        )

    elif args.command == "predict-binary":
        df = read_table(args.input)
        aligned = pipeline.align_input_dataframe(df, id_column=args.id_column)
        pred = predict_binary_classifier(
            pipeline,
            aligned,
            checkpoint_path=args.classifier,
            id_column=args.id_column,
            batch_size=args.batch_size,
        )
        write_table(pred, args.output)


if __name__ == "__main__":
    main()
