import os
import sys
import argparse
import logging
import glob
import pathlib
import mlflow
import shutil
from tqdm import tqdm

def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder containing some files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write multiplied dataset",
    )

    parser.add_argument(
        "--multiply",
        type=int,
        required=True,
        help="How much to multiply files.",
    )
    parser.add_argument(
        "--increment",
        type=int,
        required=False,
        default=0,
        help="Add to index on output files when multiplying.",
    )

    return parser


def multiply_files(file_paths, source, target, multiplication_factor, increment=0):
    files_created = 0

    for i in range(multiplication_factor):
        for entry in tqdm(file_paths):
            if os.path.isfile(entry):
                # get name of file
                file_name = pathlib.Path(os.path.basename(entry))

                # get path to the file
                full_input_dir = os.path.dirname(entry)

                # create path to the output
                rel_dir = os.path.relpath(full_input_dir, source)
                full_output_dir = os.path.join(target, rel_dir)

                # create a name for the output file
                output_file_path = os.path.join(full_output_dir, file_name.stem + f"_{i+increment}" + file_name.suffix)

                # create output dir
                if not os.path.isdir(full_output_dir):
                    logging.getLogger(__name__).info(f"Creating output subfolder {full_output_dir}")
                    os.makedirs(full_output_dir, exist_ok=True)

                #print(f"{entry} => {output_file_path}")
                if not os.path.isfile(output_file_path):
                    shutil.copy(entry, output_file_path)
                    files_created += 1

        logging.getLogger(__name__).info(f"Achieved multication {i}")

    return files_created

def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # use mpi without using mpi
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    logger.info(f"Distribution settings: rank={world_rank} size={world_size}")

    all_files_list = list(glob.glob(os.path.join(args.input, "**", "*"), recursive=True))

    logger.info(f"Total file list len={len(all_files_list)}")
    if world_rank == 0:
        mlflow.log_metric("input_file_count", len(all_files_list))

    filtered_files_list = [
        all_files_list[i] for i in range(len(all_files_list)) if i % world_size == world_rank
    ]
    logger.info(f"Process file list len={len(filtered_files_list)}")
    mlflow.log_metric("process_file_count", len(filtered_files_list))

    files_created = multiply_files(filtered_files_list, args.input, args.output, args.multiply, increment=args.increment)
    mlflow.log_metric("files_created", files_created)

    mlflow.end_run()

    logger.info("run() completed")


def main(cli_args=None):
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
