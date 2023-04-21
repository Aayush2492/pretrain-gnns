from torch_geometric.data import InMemoryDataset
import torch
import numpy as np


from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["data.csv"]

    @property
    def processed_file_names(self):
        return ["data_.pt"]

    def download(self):
        # Raw data is provided as a CSV file and is not downloaded
        pass

    def process(self):
        # nodes
        node_dict = {}

        edge_list = [
            (0, 0, 1, 1),
            (0, 0, 2, 2),
            (1, 1, 2, 2),
            (1, 1, 3, 3),
            (2, 2, 3, 3),
            (2, 2, 4, 4),
            (3, 3, 4, 4),
            (3, 3, 5, 5),
            (4, 4, 5, 5),
        ]

        for edge in edge_list:
            src, src_label, dst, dst_label = edge
            if node_dict.get(src) is None:
                node_dict[src] = src_label
            if node_dict.get(dst) is None:
                node_dict[dst] = dst_label

        node_list = list(node_dict.keys())
        node_list = [[elem, elem] for elem in node_list]
        print("xxxx", node_list)
        x = torch.tensor(np.array(node_list), dtype=torch.long)
        print("xxxx", x.shape)

        # edges
        edges_list = []
        edge_features_list = []
        for edge in edge_list:
            src, src_label, dst, dst_label = edge
            i = src
            j = dst
            edge_feature = [1, 1]  # edge label
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.id = torch.tensor([0])
#         data.y = torch.tensor([labels[i]])

        print("Data", data)

        # Apply pre-processing transformations, if any
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        # Save the processed data to disk
        # torch.save(self.collate([data]), self.processed_paths[0])
        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
