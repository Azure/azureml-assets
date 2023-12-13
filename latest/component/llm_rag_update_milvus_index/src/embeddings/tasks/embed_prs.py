# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""ParallelRunStep entrypoint for embedding data."""
import argparse
import os
import pandas as pd
import pathlib
import time
import traceback

from azureml.rag.embeddings import EmbeddingsContainer
from azureml.rag.tasks.embed import read_chunks_into_documents
from azureml.rag.utils.azureml import get_workspace_from_environment
from azureml.rag.utils.logging import (
    get_logger,
    enable_stdout_logging,
    enable_appinsights_logging,
    track_activity,
    _logger_factory
)

logger = get_logger('embed_prs')


def main(args, logger, activity_logger):
    """Embedding entrypoint for ParallelRunStep."""
    global output_data
    global embeddings_container
    global chunk_format
    output_data = args.output_data
    chunk_format = args.chunk_format

    embeddings_container = None
    if args.embeddings_container is not None:
        with track_activity(logger, 'init.load_embeddings_container') as activity_logger:
            if hasattr(activity_logger, 'activity_info'):
                activity_logger.activity_info["completionStatus"] = "Failure"
            from azureml.dataprep.fuse.dprepfuse import MountOptions, rslex_uri_volume_mount
            mnt_options = MountOptions(
                default_permission=0o555, allow_other=False, read_only=True)
            try:
                with rslex_uri_volume_mount(
                    args.embeddings_container,
                    f'{os.getcwd()}/embeddings_container', options=mnt_options
                ) as mount_context:
                    embeddings_container_dir_name = None
                    # list all folders in embeddings_container and find the latest one
                    try:
                        embeddings_container_dir_name = str(max(
                            [dir for dir in pathlib.Path(mount_context.mount_point).glob('*')
                                if dir.is_dir() and dir.name != os.environ['AZUREML_RUN_ID']],
                            key=os.path.getmtime).name)
                    except Exception as e:
                        activity_logger.warn('Failed to get latest folder from embeddings_container.')
                        logger.warn(
                            f'failed to get latest folder from {mount_context.mount_point} with {e}.')
                        pass

                    if embeddings_container_dir_name is not None:
                        logger.info(
                            f'loading from previous embeddings from {embeddings_container_dir_name}'
                            + f' in {mount_context.mount_point}')
                        try:
                            embeddings_container = EmbeddingsContainer.load(
                                embeddings_container_dir_name, mount_context.mount_point)
                            if hasattr(activity_logger, 'activity_info'):
                                activity_logger.activity_info["completionStatus"] = "Success"
                        except Exception as e:
                            activity_logger.warn('Failed to load from embeddings_container_dir_name. '
                                                 + 'Creating new Embeddings.')
                            logger.warn(
                                f'Failed to load from previous embeddings with {e}.\nCreating new Embeddings.')
            except Exception as e:
                activity_logger.warn('Failed to load from embeddings_container. Creating new Embeddings.')
                logger.warn(f'Failed to load previous embeddings from mount with {e}, '
                            + 'proceeding to create new embeddings.')

    connection_args = {}
    connection_id = os.environ.get('AZUREML_WORKSPACE_CONNECTION_ID_AOAI')
    if connection_id is not None:
        connection_args['connection_type'] = 'workspace_connection'
        connection_args['connection'] = {'id': connection_id}
    else:
        if "open_ai" in args.embeddings_model:
            ws = get_workspace_from_environment()
            connection_args["connection_type"] = "workspace_keyvault"
            connection_args["connection"] = {
                "subscription": ws.subscription_id if ws is not None else "",
                "resource_group": ws.resource_group if ws is not None else "",
                "workspace": ws.name if ws is not None else "",
                "key": "OPENAI-API-KEY"
            }

    embeddings_container = embeddings_container \
        if embeddings_container is not None \
        else EmbeddingsContainer.from_uri(args.embeddings_model, **connection_args)


def main_wrapper(args, logger):
    """Wrap main with exception handling and logging."""
    with track_activity(logger, "embed_prs") as activity_logger:
        try:
            main(args, logger, activity_logger)
        except Exception:
            # activity_logger doesn't log traceback
            activity_logger.error(f"embed_prs failed with exception: {traceback.format_exc()}")
            raise


def init():
    """Load previous embeddings if provided."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description="ParallelRunStep Agent")
    parser.add_argument("--output_data", type=str)
    parser.add_argument("--embeddings_model", type=str)
    parser.add_argument("--embeddings_container", required=False, type=str, default=None)
    parser.add_argument("--chunk_format", type=str, default="csv")
    args, _ = parser.parse_known_args()

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    enable_stdout_logging()
    enable_appinsights_logging()

    try:
        main_wrapper(args, logger)
    finally:
        if _logger_factory.appinsights:
            _logger_factory.appinsights.flush()
            time.sleep(5)


def _run_internal(mini_batch, output_data, embeddings):
    """
    Embed minibatch of chunks.

    :param mini_batch: The list of files to be processed.
    :param output_data: The output folder to save data to.
    :param embeddings: The Embeddings object that should be used to embed new data.
    """
    global chunk_format
    logger.info(f'run method start: {__file__}, run({mini_batch})')
    logger.info(f'Task id: {mini_batch.task_id}')

    # read chunks
    pre_embed = time.time()
    embeddings = embeddings.embed_and_create_new_instance(
        read_chunks_into_documents(
            (pathlib.Path(p) for p in mini_batch),
            chunk_format))
    post_embed = time.time()
    logger.info(f"Embedding took {post_embed - pre_embed} seconds")

    save_metadata = str(mini_batch.task_id) == '0'
    if save_metadata:
        logger.info('Metadata will be saved')
    else:
        logger.info('Only data will be saved')
    embeddings.save(output_data, with_metadata=save_metadata, suffix=mini_batch.task_id)


def run(mini_batch):
    """Embed minibatch of chunks."""
    global output_data
    global embeddings_container

    _run_internal(mini_batch, output_data, embeddings_container)
    return pd.DataFrame({"Files": [os.path.split(file)[-1] for file in mini_batch]})
