from typing import List, Dict
import pickle

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from sevenn.train.dataload import ASE_atoms_to_data
from sevenn.nn.node_embedding import get_type_mapper_from_specie, \
    one_hot_atom_embedding
import sevenn._keys as KEY


class AtomGraphData(torch_geometric.data.Data):
    """
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(
        self,
        x,
        edge_index,
        pos,
        edge_attr=None,
        y_energy=None,
        y_force=None,
        edge_vec=None,
        init_node_attr=True,
    ):
        super(AtomGraphData, self).__init__(
            x,
            edge_index,
            edge_attr,
            pos=pos
        )
        self[KEY.ENERGY] = y_energy
        self[KEY.FORCE] = y_force
        self[KEY.EDGE_VEC] = edge_vec
        if init_node_attr:
            self[KEY.NODE_ATTR] = x

    @staticmethod
    def data_for_E3_equivariant_model(atoms,
                                      cutoff: float,
                                      type_map: Dict[int, int]):
        """
        Args:
            atoms : 'atoms' object from ASE
            cutoff : float
            type_map : Z(atomic_number) -> one_hot index
        Returns:
            AtomGraphData for E3_equivariant_model
        """
        atomic_numbers, edge_idx, edge_vec, \
            shift, pos, cell, E, F = ASE_atoms_to_data(atoms, cutoff)
        edge_vec = torch.Tensor(edge_vec)
        edge_vec.requires_grad_(True)
        edge_idx = torch.LongTensor(edge_idx)
        pos = torch.Tensor(pos)
        F = torch.Tensor(F)

        embd = one_hot_atom_embedding(atomic_numbers, type_map)
        data = AtomGraphData(embd, edge_idx, pos,
                             y_energy=E, y_force=F,
                             edge_vec=edge_vec, init_node_attr=True)

        data[KEY.NUM_ATOMS] = len(pos)
        data.num_nodes = data[KEY.NUM_ATOMS]  # for general perpose

        avg_num_neigh = np.average(np.unique(edge_idx[0], return_counts=True)[1])
        data[KEY.AVG_NUM_NEIGHBOR] = avg_num_neigh
        return data

    def to_dict(self):
        return {k: v for k, v in self.items()}
