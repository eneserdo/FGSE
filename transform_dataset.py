# After running the prepare_dataset.py, we will have a processed folder with all the data points
# If you want to represent sequences of graphs as monographs, run the following code

import argparse
import glob
import os
import pathlib
import shutil

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from dataset import get_processing_dir


def transform_datapoint(datapoint_path):
    # It will convert the datapoint to monograph
    # This is actually very similar to flatten_scene_graphs function in bimanual's original code
    # This function is not sensitive to multiple occurences of the same object as in the original code
 
    data = torch.load(datapoint_path)
    # data=[Data(x=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), edge_index=torch.tensor([[0, 1], [1, 2]]).T, edge_attr=torch.tensor([[0,0,1,1], [1,1,2,2]]), y=torch.tensor([1])),
    #       Data(x=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), edge_index=torch.tensor([[0, 2], [1, 2]]).T, edge_attr=torch.tensor([[0,0,2,2], [1,1,2,2]]), y=torch.tensor([2])),
    #       Data(x=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), edge_index=torch.tensor([[0, 1], [0, 2]]).T, edge_attr=torch.tensor([[0,0,1,1], [0,0,2,2]]), y=torch.tensor([3]))
    # ]
          
    monograph_x = []
    node_counts = []

    # create a dict for global idx to node idx
    local2global_idx = {}
    global_idx = 0
    
    # Nodes
    for graph_idx, graph in enumerate(data):
        node_counts.append(graph.x.shape[0])

        for node_idx, node in enumerate(graph.x):
            local2global_idx[(graph_idx, node_idx)] = global_idx
            monograph_x.append(node)
            global_idx += 1

    monograph_x = torch.stack(monograph_x)

    # Edges between nodes of the same graph
    monograph_edge_index = []
    monograph_edge_attr = []

    # graph.edge_index: [2, E]
    # graph.edge_attr: [E, D]

    for graph_idx, graph in enumerate(data):
        for edge_idx, edge in enumerate(graph.edge_index.t()):
            l_node1, l_node2 = edge.tolist()
            g_node1=local2global_idx[(graph_idx, l_node1)]
            g_node2=local2global_idx[(graph_idx, l_node2)]
            
            attribute = graph.edge_attr[edge_idx]
            attribute = torch.cat([attribute, torch.zeros(1)], dim=0) # for temporal edge

            monograph_edge_index.append([g_node1, g_node2])
            monograph_edge_attr.append(attribute)
        
    n_spatial_attr = monograph_edge_attr[0].shape[0] - 1
    temporal_edge_attr = torch.tensor([0]*n_spatial_attr + [1])

    # Edges between nodes of consecutive graphs
    for graph_idx in range(1, len(data)-1):
        graph_prev = data[graph_idx-1]
        graph_curr = data[graph_idx]

        for pnidx, pnode in enumerate(graph_prev.x):
            for cnidx, cnode in enumerate(graph_curr.x):
                
                # Burda akıllı bir şekilde birden fazla aynı obje varsa ne yapmalıyız?
                # if len(graph_prev.x) == len(graph_curr.x):
                #     if pnidx == cnidx:
                #         ...
                #         # Bu kesin ok
                #     else:
                #         ...
                #         # Bu hiç vaki mi?
                # else:
                #     ...
                #     # burda hepsini bağla

                
                # yada paperdaki gibi herşeyi bağla
                # Check if they are the same node
                if torch.all(pnode == cnode):
                    g_pnode = local2global_idx[(graph_idx-1, pnidx)]
                    g_cnode = local2global_idx[(graph_idx, cnidx)]

                    monograph_edge_index.append([g_pnode, g_cnode])
                    monograph_edge_attr.append(temporal_edge_attr)
                    # symmetric
                    monograph_edge_index.append([g_cnode, g_pnode])
                    monograph_edge_attr.append(temporal_edge_attr)


    monograph_edge_index = torch.tensor(monograph_edge_index).t().contiguous()
    monograph_edge_attr = torch.stack(monograph_edge_attr)

    # labels
    y = [] 
    for graph_idx, graph in enumerate(data):
        y.append(graph.y)
    
    y = torch.stack(y)
    
    new_data = Data(x=monograph_x, edge_index=monograph_edge_index, edge_attr=monograph_edge_attr, y=y, node_counts=torch.tensor(node_counts))

    torch.save(new_data, datapoint_path)
    

def transform_dataset(processed_dir: pathlib.Path):
    # it will read the processed folder and convert each datapoint to monograph
    
    print("Copying the processed folder...")
    monograph_dir = processed_dir.name + "_G"
    shutil.copytree(processed_dir, monograph_dir)
    
    # Traverse the new folder and for each datapoint, convert it to monograph
    data_paths = glob.glob(os.path.join(monograph_dir, "subject_*", "*.pt"))
    
    for data_path in tqdm(data_paths):
        transform_datapoint(data_path)
        


def arg_parser():
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("-rt", "--root", type=str, default=".", help="Root directory of dataset.")
    
    # to get the processed folder's name
    ap.add_argument("-tl", "--temporal_length", type=int, required=True, help="Temporal length of the input (if -1, processes whole video)")
    ap.add_argument("-ds", "--downsample", type=int, default=1, help="Downsampling ratio")
    ap.add_argument("-vf", "--use_vf", action="store_true", help="Use visual features")
    ap.add_argument("-fd", "--filtered_data", action="store_true", help="Use filtered data")
    
    return ap.parse_args()


def main():
    print("Transforming dataset...")
    args = arg_parser()

    # check if it is already ready
    pr_dir = pathlib.Path(get_processing_dir(args.root, args.temporal_length, args.downsample, args.use_vf, args.filtered_data, False))

    assert pr_dir.exists(), f"The dataset is not processed in {pr_dir}\n, (Run prepare_dataset.py first)"
        
    transform_dataset(pr_dir)



if __name__ == "__main__":
    main()