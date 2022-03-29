from src.util.constants import IN_MEMORY, ON_DISK


def get_number_of_nodes(dataset, opt):
    dataset_loading_location = opt.get('dataset_loading_location')

    if dataset_loading_location == ON_DISK:
        return dataset[0].x.size()[1]
    elif dataset_loading_location == IN_MEMORY:
        return dataset[0][0].size()[1]
    else:
        raise ValueError(f"Invalid dataset loading location {dataset_loading_location}")
