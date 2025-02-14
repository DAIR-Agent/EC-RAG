# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Command line interface for the query module."""

import os
from pathlib import Path
from typing import cast

import pandas as pd
from graphrag.query.llm.base import BaseLLMCallback

from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)
from graphrag.index.progress import PrintProgressReporter
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.vector_stores import VectorStoreFactory, VectorStoreType

from .factories import get_global_search_engine, get_local_search_engine
from .indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)

from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)

reporter = PrintProgressReporter("")


def __get_embedding_description_store(
    vector_store_type: str = VectorStoreType.LanceDB, config_args: dict | None = None
):
    """Get the embedding description store."""
    if not config_args:
        config_args = {}

    config_args.update({
        "collection_name": config_args.get(
            "query_collection_name",
            config_args.get("collection_name", "description_embedding"),
        ),
    })

    description_embedding_store = VectorStoreFactory.get_vector_store(
        vector_store_type=vector_store_type, kwargs=config_args
    )

    description_embedding_store.connect(**config_args)
    return description_embedding_store


def run_global_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
    conversation_history: ConversationHistory,
    message: str,
    callback: BaseLLMCallback,
):
    """Run a global search with the given query."""

    splitter = "*"
    if splitter in  data_dir:
        data_dir_list = data_dir.split(splitter)
    else:
        data_dir_list = [data_dir]

    final_nodes_list = []
    final_community_reports_list = []
    final_entities_list = []
    entities_list = []
    reports_list = []

    for data_dir in data_dir_list:

        data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
        data_path = Path(data_dir)

        final_nodes: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_nodes.parquet"
        )
        final_nodes_list.append(final_nodes)

        try:
            final_entities: pd.DataFrame = pd.read_parquet(
                data_path / "create_final_entities_ori.parquet"
            )
        except:
            final_entities: pd.DataFrame = pd.read_parquet(
                data_path / "create_final_entities.parquet"
            )

        final_entities_list.append(final_entities)

        final_community_reports: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )
        final_community_reports_list.append(final_community_reports)

        reports = read_indexer_reports(
            final_community_reports, final_nodes, community_level
        )
        reports_list.append(reports)
        
        entities = read_indexer_entities(final_nodes, final_entities, community_level)
        entities_list.append(entities)

    context_chunks_list = []
    context_records_list = []
    for i in range(len(data_dir_list)):
        search_engine = get_global_search_engine(
            config,
            reports=reports_list[i],
            entities=entities_list[i],
            response_type=response_type,
        )
        context_chunks, context_records = search_engine.get_context(conversation_history=conversation_history)
        context_chunks_list = context_chunks_list + context_chunks
        context_records_list.append(context_records["reports"].drop_duplicates())

    reports = []
    for i in range(len(reports_list)):
        reports = reports + reports_list[i]
    entities = []
    for i in range(len(entities_list)):
        entities = entities + entities_list[i]

    context_chunks = context_chunks_list
    context_records = {}
    context_records["reports"] = pd.concat(context_records_list)

    # print("\n\n\n\n".join(context_chunks))
    # assert(False)
    
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities,
        response_type=response_type,
    )

    result = search_engine.search_with_context(query=query, context_chunks=context_chunks, context_records=context_records, conversation_history=conversation_history, callback=callback)

    reporter.success(f"Global Search Response: {result.response}")
    return result.response


def run_local_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
    conversation_history: ConversationHistory,
    message: str,
    callback: BaseLLMCallback,
):
    """Run a local search with the given query."""

    ### multi data dir
    splitter = "*"
    if splitter in  data_dir:
        data_dir_list = data_dir.split(splitter)
    else:
        data_dir_list = [data_dir]

    final_nodes_list = []
    final_community_reports_list = []
    final_text_units_list = []
    final_relationships_list = []
    final_relationships_list_extra = []
    final_entities_list = []
    final_entities_list_extra = []
    final_covariates_list = []
    entities_list = []
    covariates_list = []
    reports_list = []
    description_embedding_store_list = []
    entities_extra_list = []

    for data_dir in data_dir_list:

        data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
        data_path = Path(data_dir)

        final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
        final_nodes_list.append(final_nodes)

        final_community_reports = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )
        final_community_reports_list.append(final_community_reports)

        final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
        final_text_units_list.append(final_text_units)

        try:
            final_relationships = pd.read_parquet(
                data_path / "create_final_relationships_ori.parquet"
            )
        except:
            final_relationships = pd.read_parquet(
                data_path / "create_final_relationships.parquet"
            )

        final_relationships_expanded = pd.read_parquet(
            data_path / "create_final_relationships.parquet"
        )
        final_relationships = final_relationships[final_relationships.apply(lambda row: len(final_relationships_expanded[(final_relationships_expanded["source"]==row['source']) & (final_relationships_expanded["target"]==row['target'])])>0, axis=1)]
        # final_relationships_extra = final_relationships_expanded[final_relationships_expanded.apply(lambda row: len(final_relationships[(final_relationships["source"]==row['source']) & (final_relationships["target"]==row['target'])])==0, axis=1)]
        final_relationships_extra = final_relationships_expanded
        final_relationships_list.append(final_relationships)
        final_relationships_list_extra.append(final_relationships_extra)

        # final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
        # final_nodes_list.append(final_nodes)

        try:
            final_entities = pd.read_parquet(data_path / "create_final_entities_ori.parquet")
        except:
            final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")

        final_entities_expanded = pd.read_parquet(data_path / "create_final_entities.parquet")
        final_entities = final_entities[final_entities["name"].apply(lambda x: x in final_entities_expanded["name"].values)]
        # final_entities_extra = final_entities_expanded[final_entities_expanded["name"].apply(lambda x: x not in final_entities["name"].values)]
        final_entities_extra = final_entities_expanded
        final_entities_list.append(final_entities)
        final_entities_list_extra.append(final_entities_extra)

        final_covariates_path = data_path / "create_final_covariates.parquet"
        final_covariates = (
            pd.read_parquet(final_covariates_path)
            if final_covariates_path.exists()
            else None
        )
        if final_covariates != None:
            final_covariates_list.append(final_covariates)

        vector_store_args = (
            config.embeddings.vector_store if config.embeddings.vector_store else {}
        )
        vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

        description_embedding_store = __get_embedding_description_store(
            vector_store_type=vector_store_type,
            config_args=vector_store_args,
        )

        entities = read_indexer_entities(final_nodes, final_entities, community_level)
        entities_extra_list.append(final_entities_extra)
        entities_list.append(entities)

        store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store
        )
        description_embedding_store_list.append(description_embedding_store)

        covariates = (
            read_indexer_covariates(final_covariates)
            if final_covariates is not None
            else []
        )
        covariates_list.append(covariates)

        reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        )
        reports_list.append(reports)

    context_text_list = []

    reports_list_context_records = []
    relationships_list_context_records = []
    claims_list_context_records = []
    entities_list_context_records = []
    for i in range(len(data_dir_list)):
        search_engine = get_local_search_engine(
            config,
            reports=read_indexer_reports(
                final_community_reports_list[i], final_nodes_list[i], community_level
            ),
            text_units=read_indexer_text_units(final_text_units_list[i]),
            entities=entities_list[i],
            entities_extra=entities_extra_list[i],
            relationships=read_indexer_relationships(final_relationships_list[i]),
            covariates={"claims": covariates_list[i]},
            description_embedding_store=description_embedding_store_list[i],
            response_type=response_type,
            data_path = data_path,
        )
        context_text, context_records = search_engine.get_context(query=query, conversation_history=conversation_history)
        context_text_list.append(context_text)
        if 'reports' in context_records.keys():
            reports_list_context_records.append(context_records['reports'])
        if 'relationships' in context_records.keys():   
            relationships_list_context_records.append(context_records['relationships'])
        if 'claims' in context_records.keys(): 
            claims_list_context_records.append(context_records['claims'])
        if 'entities' in context_records.keys(): 
            entities_list_context_records.append(context_records['entities'])

    context_records = {}
    if len(reports_list_context_records) > 0:
        context_records["reports"] = pd.concat(reports_list_context_records)
    if len(reports_list_context_records) > 0:
        context_records["relationships"] = pd.concat(reports_list_context_records)
    if len(claims_list_context_records) > 0:
        context_records["claims"] = pd.concat(claims_list_context_records)
    if len(entities_list_context_records) > 0:
        context_records["entities"] = pd.concat(entities_list_context_records)

    context_text = "\n\n".join(context_text_list)
    result = search_engine.search_with_context(query=query, context_text=message+"\n"+context_text, context_records=context_records, conversation_history=conversation_history, callback=callback)
    reporter.success(f"Local Search Response: {result.response}")
    return result.response


def _configure_paths_and_settings(
    data_dir: str | None, root_dir: str | None
) -> tuple[str, str | None, GraphRagConfig]:
    if data_dir is None and root_dir is None:
        msg = "Either data_dir or root_dir must be provided."
        raise ValueError(msg)
    if data_dir is None:
        data_dir = _infer_data_dir(cast(str, root_dir))
    config = _create_graphrag_config(root_dir, data_dir)
    return data_dir, root_dir, config


def _infer_data_dir(root: str) -> str:
    output = Path(root) / "output"
    # use the latest data-run folder
    if output.exists():
        folders = sorted(output.iterdir(), key=os.path.getmtime, reverse=True)
        if len(folders) > 0:
            folder = folders[0]
            return str((folder / "artifacts").absolute())
    msg = f"Could not infer data directory from root={root}"
    raise ValueError(msg)


def _create_graphrag_config(root: str | None, data_dir: str | None) -> GraphRagConfig:
    """Create a GraphRag configuration."""
    return _read_config_parameters(cast(str, root or data_dir))


def _read_config_parameters(root: str):
    _root = Path(root)
    settings_yaml = _root / "settings.yaml"
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    settings_json = _root / "settings.json"

    if settings_yaml.exists():
        reporter.info(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("r") as file:
            import yaml

            data = yaml.safe_load(file)
            return create_graphrag_config(data, root)

    if settings_json.exists():
        reporter.info(f"Reading settings from {settings_json}")
        with settings_json.open("r") as file:
            import json

            data = json.loads(file.read())
            return create_graphrag_config(data, root)

    reporter.info("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)
