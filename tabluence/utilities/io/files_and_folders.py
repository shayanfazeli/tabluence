import os


def clean_folder(folder_path: str) -> None:
    """
    Giving the option to keep or erase

    Parameters
    ----------
    folder_path: `str`, required
        The folder path to process
    """
    if os.path.isdir(folder_path):
        choice = input(
            f"\n\t ?? the representation folder already exists at {folder_path}. delete and recreate (y) or continue(n)?\n")
        if choice.lower() == 'y':
            os.system(f'rm -rf {folder_path}')
            os.makedirs(os.path.abspath(folder_path))
