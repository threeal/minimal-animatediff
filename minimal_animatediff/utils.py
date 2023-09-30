import os

from huggingface_hub import snapshot_download


def populate_snapshot(path: str) -> str:
    """
    Populates the snapshot of a specified path from the 'threeal/AnimateDiffMirrors' repository.

    Args:
        path (str): The path to the file or directory in the repository to be populated.

    Returns:
        str: The local path of the populated snapshot.

    Example:
        >>> populate_snapshot("MotionModules/mm_sd_v15.ckpt")
        'snapshots/MotionModules/mm_sd_v15.ckpt'
    """
    snapshot_download(
        repo_id="threeal/AnimateDiffMirrors",
        local_dir="snapshots",
        allow_patterns=[path],
    )
    return os.path.join("snapshots", path)
