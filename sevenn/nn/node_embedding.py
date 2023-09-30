from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional
from ase.symbols import symbols2numbers
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


#TODO: put this to model_build and do not preprocess data by onehot
@compile_mode('script')
class OnehotEmbedding(nn.Module):
    """
    x : tensor of shape (N, 1)
    x_after : tensor of shape (N, num_classes)
    It overwrite data_key_x
    and saves input to data_key_save and output to data_key_additional
    I know this is strange but it is for compatibility with previous version
    and to specie wise shift scale work
    ex) [0 1 1 0] -> [[1, 0] [0, 1] [0, 1] [1, 0]] (num_classes = 2)
    """
    def __init__(
        self,
        num_classes: int,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_save: str = KEY.ATOM_TYPE,
        data_key_additional: str = KEY.NODE_ATTR,  # additional output
    ):
        super().__init__()
        self.num_classes = num_classes
        self.KEY_X = data_key_x
        self.KEY_SAVE = data_key_save
        self.KEY_ADDITIONAL = data_key_additional

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        inp = data[self.KEY_X]
        embd = torch.nn.functional.one_hot(inp, self.num_classes)
        embd = embd.float()
        data[self.KEY_X] = embd
        if self.KEY_ADDITIONAL is not None:
            data[self.KEY_ADDITIONAL] = embd
        if self.KEY_SAVE is not None:
            data[self.KEY_SAVE] = inp
        return data


def get_type_mapper_from_specie(specie_list: List[str]):
    """
    from ['Hf', 'O']
    return {72: 0, 16: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


# deprecated
def one_hot_atom_embedding(atomic_numbers: List[int], type_map: Dict[int, int]):
    """
    atomic numbers from ase.get_atomic_numbers
    type_map from get_type_mapper_from_specie()
    """
    num_classes = len(type_map)
    try:
        type_numbers = torch.LongTensor([type_map[num] for num in atomic_numbers])
    except KeyError as e:
        raise ValueError(f"Atomic number {e.args[0]} is not expected")
    embd = torch.nn.functional.one_hot(type_numbers, num_classes)
    embd = embd.to(torch.get_default_dtype())

    return embd


def main():
    _ = 1


if __name__ == "__main__":
    main()

