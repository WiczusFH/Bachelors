import os

# Remember to change f"{var:0%d}"
INDEX_LEN = 2


def is_index_valid(object_name, format_name, starting_index):
    return int(object_name[:-len(format_name)][-INDEX_LEN:]) >= starting_index


def add_index2path(target_file, target_format, add_format=False, index_present=False, index=0):
    formatless_path = target_file.split('.')[0]
    if not index_present:
        indexed_formatless_path = formatless_path + f"{index:02}"
    else:
        indexed_formatless_path = formatless_path
    while os.path.exists(f"{indexed_formatless_path}{target_format}"):
        index += 1
        indexed_formatless_path = indexed_formatless_path[:-INDEX_LEN] + f"{index:02}"

    if add_format:
        return f"{indexed_formatless_path}{target_format}"
    return indexed_formatless_path


class Iterator:
    def __init__(self, source_folder, target_folder, flatten=False):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.object_types = os.listdir(source_folder)
        self.flatten = flatten

    def execute(self, object_action, source_object_predicate=lambda x: True):
        for object_type in self.object_types:
            source_train_folder = os.path.join(self.source_folder, object_type, "train")
            if self.flatten:
                target_train_folder = self.target_folder
            else:
                target_train_folder = os.path.join(self.target_folder, object_type, "train")
            if not os.path.exists(target_train_folder):
                os.makedirs(target_train_folder)
            object_names = [name for name in os.listdir(source_train_folder) if source_object_predicate(name)]
            for object_name in object_names:
                source_path = os.path.join(source_train_folder, object_name)
                target_path = os.path.join(target_train_folder, object_name)
                if self.flatten:
                    object_action(source_path, target_path, object_name)
                else:
                    object_action(source_path, target_path, object_name)
