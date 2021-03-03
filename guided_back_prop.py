def create_maps_folders(main_folder, beat, labels, delete_prior):
    if delete_prior and Path(main_folder).exists():
        shutil.rmtree(main_folder)
    for label in labels:
        folder = Path(main_folder) / f"label_{beat}_beat/"
        Path(folder / label).mkdir(parents=True, exist_ok=True)
    return folder