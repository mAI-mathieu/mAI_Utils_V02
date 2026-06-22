from pathlib import Path, PurePosixPath, PureWindowsPath


def save_text_file(text: str, file_name: str, folder: str, overwrite: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")
    if file_name.strip() == "":
        raise ValueError("file_name must be a non-empty string")
    if not isinstance(folder, str):
        raise TypeError("folder must be a string")

    _validate_simple_file_name(file_name)

    folder_path = Path.cwd() if folder == "" else Path(folder)
    if "\0" in folder:
        raise ValueError("folder contains an unsafe null character")
    if folder_path.exists() and not folder_path.is_dir():
        raise ValueError(f"folder must be a directory path: {folder}")

    folder_path.mkdir(parents=True, exist_ok=True)
    output_path = folder_path / file_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite is disabled: {output_path}")

    output_path.write_text(text, encoding="utf-8")
    return str(output_path.resolve())


def _validate_simple_file_name(file_name: str) -> None:
    if "\0" in file_name:
        raise ValueError("file_name contains an unsafe null character")
    if "/" in file_name or "\\" in file_name:
        raise ValueError("file_name must be a simple file name, not a path")
    if file_name in {".", ".."}:
        raise ValueError("file_name must not be a path traversal value")

    windows_path = PureWindowsPath(file_name)
    posix_path = PurePosixPath(file_name)
    if windows_path.is_absolute() or posix_path.is_absolute():
        raise ValueError("file_name must not be an absolute path")
    if windows_path.drive or windows_path.root or posix_path.root:
        raise ValueError("file_name must be a simple file name, not a path")
