import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


"""
Address Parsing with Named Entity Recognition (NER)

This module provides optimized tools for parsing addresses using HuggingFace transformer models
with Named Entity Recognition. It offers both simple and advanced processing approaches to handle
datasets with varying address lengths and complexity.

Key Features:
    - Single-stage processing for uniform datasets
    - Two-stage processing optimized for mixed-length addresses  
    - Automatic GPU/CPU device detection and management
    - Batch processing with configurable sizes
    - Progress tracking and comprehensive result summaries
    - Structured output conversion for analysis

Main Functions:
    parse_addresses_basic(): Simple single-stage processing with fixed batch size.
        Best for small datasets, debugging, or uniform address lengths.
    
    parse_addresses_pipeline(): Advanced two-stage processing with dynamic batching.
        Optimized for production use with large datasets containing mixed address lengths.
        Automatically separates short and long addresses for optimal memory usage.
    
    convert_to_entity_dataframe(): Converts parsing results into structured DataFrame
        format suitable for analysis or database storage.

Processing Strategies:
    - Short addresses (≤ token_threshold): Processed with large batch sizes for high throughput
    - Long addresses (> token_threshold): Processed with smaller batches to avoid memory issues
    - Automatic tokenization analysis to determine optimal processing approach

Dependencies:
    - pandas: DataFrame operations and data handling
    - torch: PyTorch backend for model inference
    - transformers: HuggingFace model pipeline and tokenization
    - tqdm: Progress bar visualization
    - typing: Type hints for better code documentation

Example Usage:
    Basic processing:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'address': ['14 Barnsbury Road', 'Flat 14a, 14 Barnsbury Road']})
    >>> results = parse_addresses_basic(df, 'path/to/model')
    >>> print(f"Parsed {results['summary']['successful_parses']} addresses")
    
    Advanced processing for large datasets:
    >>> results = parse_addresses_pipeline(
    ...     df=large_address_df,
    ...     model_path='./address-ner-model',
    ...     short_batch_size=4096,  # High throughput for short addresses
    ...     long_batch_size=32,     # Conservative for long addresses
    ...     token_threshold=128
    ... )
    
    Convert to structured format:
    >>> entity_df = convert_to_entity_dataframe(results)
    >>> print(entity_df.groupby('label').size())  # Count entities by type

Performance Considerations:
    - GPU processing significantly faster than CPU for large datasets
    - Two-stage processing prevents out-of-memory errors with mixed-length data
    - Batch size tuning critical for optimal throughput vs memory usage
    - Progress tracking helps monitor processing of large datasets

Output Format:
    Results contain both summary statistics and detailed entity information:
    - Summary: Processing metrics, success rates, batch sizes used
    - Results: Per-address parsing with entities, confidence scores, and positions
    - Entities: Structured format with type, text, position, and confidence data
"""

# ==================== SHARED UTILITIES ====================


def format_result(
    row_index: int, address: str, datapoint_id, predictions: List[Dict]
) -> Dict:
    """
    Convert HuggingFace NER predictions to standardized result format.

    Args:
        row_index: Index of the address in the original dataset
        address: Original address string
        datapoint_id: Unique identifier for this address
        predictions: List of HuggingFace NER predictions

    Returns:
        Standardized result dictionary
    """
    # Convert HF format to our format
    entities = []
    for pred in predictions:
        entities.append(
            {
                "type": pred["entity_group"],
                "text": pred["word"],
                "start": pred["start"],
                "end": pred["end"],
                "confidence": pred["score"],
            }
        )

    # Group by type
    parsed_components = {}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in parsed_components:
            parsed_components[entity_type] = []
        parsed_components[entity_type].append(entity["text"])

    return {
        "row_index": row_index,
        "datapoint_id": datapoint_id,
        "original_address": address,
        "entities": entities,
        "parsed_components": parsed_components,
    }


def _create_pipeline(model_path: str, device: int, batch_size: int):
    """Create HuggingFace pipeline with standard configuration."""
    return pipeline(
        "token-classification",
        model=model_path,
        device=device,
        batch_size=batch_size,
        aggregation_strategy="simple",
        torch_dtype=torch.float16,
    )


def _prepare_data(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List]:
    """Extract and prepare address data from DataFrame."""
    addresses = df[target_column].fillna("").astype(str).tolist()
    datapoint_ids = df.get("datapoint_id", df.index).tolist()
    return addresses, datapoint_ids


def _calculate_summary(addresses: List[str], results: List[Dict], **kwargs) -> Dict:
    """Calculate processing summary statistics."""
    successful_parses = sum(1 for r in results if r and r["entities"])

    summary = {
        "total_addresses": len(addresses),
        "successful_parses": successful_parses,
        "failed_parses": len(addresses) - successful_parses,
        "success_rate": successful_parses / len(addresses) if addresses else 0,
    }
    summary.update(kwargs)  # Add any additional metrics
    return summary


# ==================== MAIN FUNCTIONS ====================


def parse_addresses_basic(
    df: pd.DataFrame,
    model_path: str,
    target_column: str = "address",
    batch_size: int = 512,
    device: Optional[int] = None,
    show_progress: bool = True,
) -> Dict:
    """
    Parse addresses using single-stage HuggingFace pipeline processing.

    This is the simple, straightforward approach that processes all addresses
    with the same batch size. Good for:
    - Debugging and testing
    - Small datasets
    - When addresses are relatively uniform in length
    - Quick prototyping

    For production use with mixed-length addresses, consider parse_addresses_pipeline()
    which uses optimized two-stage processing.

    Args:
        df: DataFrame containing addresses to parse
        model_path: Path to HuggingFace model (local or hub)
        target_column: Column name containing addresses
        batch_size: Fixed batch size for all processing
        device: GPU device (-1 for CPU, 0+ for GPU). Auto-detected if None
        show_progress: Whether to show progress bar

    Returns:
        Dict with 'summary' and 'results' keys containing parsing results
    """
    # Auto-detect device
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    # Prepare data
    addresses, datapoint_ids = _prepare_data(df, target_column)

    print(
        f"Loading model and processing {len(addresses)} addresses with batch_size={batch_size}"
    )

    # Create pipeline
    nlp = _create_pipeline(model_path, device, batch_size)

    # Process all addresses
    all_predictions = nlp(addresses)

    # Convert results
    results = []
    iterator = zip(addresses, all_predictions, datapoint_ids)
    if show_progress:
        iterator = tqdm(iterator, total=len(addresses), desc="Converting results")

    for i, (address, predictions, datapoint_id) in enumerate(iterator):
        result = format_result(i, address, datapoint_id, predictions)
        results.append(result)

    # Calculate summary
    summary = _calculate_summary(addresses, results, batch_size_used=batch_size)

    return {"summary": summary, "results": results}


def parse_addresses_pipeline(
    df: pd.DataFrame,
    model_path: str,
    target_column: str = "address",
    short_batch_size: int = 2048,
    long_batch_size: int = 32,
    token_threshold: int = 128,
    device: Optional[int] = None,
) -> Dict:
    """
    Parse addresses using optimized two-stage processing for mixed-length data.

    This approach analyzes address lengths and processes them in two optimized stages:
    - Short addresses (≤ token_threshold): Large batches for high throughput
    - Long addresses (> token_threshold): Small batches to avoid memory issues

    Recommended for:
    - Production environments
    - Large datasets with varying address lengths
    - Memory-constrained environments
    - When processing speed is important

    Args:
        df: DataFrame containing addresses to parse
        model_path: Path to HuggingFace model (local or hub)
        target_column: Column name containing addresses
        short_batch_size: Batch size for short addresses (larger = faster)
        long_batch_size: Batch size for long addresses (smaller = less memory)
        token_threshold: Token count threshold to split short/long addresses
        device: GPU device (-1 for CPU, 0+ for GPU). Auto-detected if None

    Returns:
        Dict with 'summary' and 'results' keys containing parsing results

    Example:
        >>> results = parse_addresses_pipeline(
        ...     df=address_df,
        ...     model_path="./my-ner-model",
        ...     short_batch_size=4096,  # High throughput for short addresses
        ...     long_batch_size=16,     # Conservative for long addresses
        ...     token_threshold=100
        ... )
    """
    device = device or (0 if torch.cuda.is_available() else -1)

    # Prepare data
    addresses, datapoint_ids = _prepare_data(df, target_column)

    # Load tokenizer and split data by length
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    short_data, long_data = _split_by_length(
        addresses, datapoint_ids, tokenizer, token_threshold
    )

    print(f"Short addresses: {len(short_data):,} | Long addresses: {len(long_data):,}")

    # Process each stage
    results = [None] * len(addresses)
    _process_stage(short_data, "short", short_batch_size, model_path, device, results)
    _process_stage(long_data, "long", long_batch_size, model_path, device, results)

    # Calculate summary with two-stage metrics
    summary = _calculate_summary(
        addresses,
        results,
        short_addresses=len(short_data),
        long_addresses=len(long_data),
        short_batch_size=short_batch_size,
        long_batch_size=long_batch_size,
        token_threshold=token_threshold,
    )

    return {"summary": summary, "results": results}


# ==================== TWO-STAGE PROCESSING HELPERS ====================


def _split_by_length(
    addresses: List[str], datapoint_ids: List, tokenizer, token_threshold: int
) -> Tuple[List[Tuple], List[Tuple]]:
    """Split addresses into short and long categories based on token count."""
    short_data, long_data = [], []

    for i, addr in enumerate(tqdm(addresses, desc="Analyzing address lengths")):
        token_count = len(tokenizer.tokenize(addr))
        data_tuple = (i, addr, datapoint_ids[i])

        if token_count <= token_threshold:
            short_data.append(data_tuple)
        else:
            long_data.append(data_tuple)

    return short_data, long_data


def _process_stage(
    stage_data: List[Tuple],
    stage_name: str,
    batch_size: int,
    model_path: str,
    device: int,
    results: List,
) -> None:
    """
    Process one stage (short or long addresses) with optimized batch size.

    Args:
        stage_data: List of (index, address, datapoint_id) tuples
        stage_name: "short" or "long" for logging
        batch_size: Batch size optimized for this stage
        model_path: Path to the model
        device: Device to use
        results: List to store results (modified in-place)
    """
    if not stage_data:
        print(f"No {stage_name} addresses to process")
        return

    print(
        f"Processing {len(stage_data):,} {stage_name} addresses (batch_size={batch_size})..."
    )

    # Create pipeline for this stage
    pipeline_instance = _create_pipeline(model_path, device, batch_size)

    # Extract data for processing
    indices, addresses_list, datapoint_ids = zip(*stage_data)

    # Process all addresses in this stage
    predictions = pipeline_instance(list(addresses_list))

    # Store results in the correct positions
    for idx, addr, datapoint_id, preds in zip(
        indices, addresses_list, datapoint_ids, predictions
    ):
        results[idx] = format_result(idx, addr, datapoint_id, preds)

    print(f"✓ Completed {len(stage_data):,} {stage_name} addresses")


# ==================== RESULT PROCESSING ====================
