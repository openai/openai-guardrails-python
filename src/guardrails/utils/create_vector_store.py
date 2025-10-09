"""Vector store creation helper script.

This script allows users to create vector stores from files or directories
and get the vector store ID for use with the anti-hallucination guardrail.

Usage:
    python create_vector_store.py /path/to/documents
    python create_vector_store.py /path/to/single/file.pdf
"""

import asyncio
import logging
import sys
from pathlib import Path

from openai import AsyncOpenAI

from .._openai_utils import prepare_openai_kwargs

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_FILE_TYPES = {
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".doc",
    ".docx",
    ".go",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".pdf",
    ".php",
    ".pptx",
    ".py",
    ".rb",
    ".sh",
    ".tex",
    ".ts",
    ".txt",
}


async def create_vector_store_from_path(
    path: str | Path,
    client: AsyncOpenAI,
) -> str:
    """Create a vector store from a file or directory path.

    Args:
        path: Path to file or directory containing documents.
        client: OpenAI client instance.

    Returns:
        Vector store ID.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If no supported files are found.
        Exception: For other OpenAI API errors.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    try:
        # Create vector store
        logger.info(f"Creating vector store from path: {path}")
        vector_store = await client.vector_stores.create(name=f"anti_hallucination_{path.name}")

        # Get list of files to upload
        file_paths = []
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES:
            file_paths = [path]
        elif path.is_dir():
            file_paths = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_FILE_TYPES]

        if not file_paths:
            raise ValueError(f"No supported files found in {path}")

        logger.info(f"Found {len(file_paths)} files to upload")

        # Upload files
        file_ids = []
        for file_path in file_paths:
            try:
                with open(file_path, "rb") as f:
                    file_result = await client.files.create(file=f, purpose="assistants")
                    file_ids.append(file_result.id)
                    logger.info(f"Uploaded: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to create file {file_path}: {e}")

        if not file_ids:
            raise ValueError("No files were successfully uploaded")

        # Add files to vector store
        logger.info("Adding files to vector store...")
        for file_id in file_ids:
            await client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file_id)

        # Wait for files to be processed
        logger.info("Waiting for files to be processed...")
        while True:
            files = await client.vector_stores.files.list(vector_store_id=vector_store.id)

            # Check if all files are completed
            statuses = [file.status for file in files.data]
            if all(status == "completed" for status in statuses):
                logger.info(f"Vector store created successfully: {vector_store.id}")
                return vector_store.id
            elif any(status == "error" for status in statuses):
                raise Exception("Some files failed to process")

            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in create_vector_store_from_path: {e}")
        raise


async def main():
    """Main function to create a vector store from command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python create_vector_store.py <path_to_documents>")
        print("Example: python create_vector_store.py /path/to/documents")
        sys.exit(1)

    path = sys.argv[1]

    try:
        client = AsyncOpenAI(**prepare_openai_kwargs({}))
        vector_store_id = await create_vector_store_from_path(path, client)

        print("\n✅ Vector store created successfully!")
        print(f"Vector Store ID: {vector_store_id}")
        print("\nUse this ID in your anti-hallucination guardrail config:")
        print(f'{{"knowledge_source": "{vector_store_id}"}}')

    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
