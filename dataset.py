import copy
import glob
import json
import os
import pathlib
import pickle
import shutil
import subprocess
from collections import Counter
from os.path import join as opj
from typing import Callable, List, Tuple

import enlighten
import natsort
import numpy as np
import torch
from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.data import download_url, extract_zip

# Global variables
OBJECTS = ["screwdriver", "hose", "valve", "valve_terminal", "box_w_screws", 
           "box_w_membrane", "soldering_station", "soldering_iron", "soldering_tin", 
           "soldering_board", "capacitor", "robot", "human", "RightHand", "eef_robot", 
           "robot_base"]

RELATIONS = ['contact', 'above', 'below', 'left of', 'right of', 'behind of', 'in front of',
             'inside', 'surround', 'moving together', 'halting together', 'fixed moving together',
             'getting close', 'moving apart', 'stable']

_ACTIONS = ['approach', 'grab', 'plug', 'join', 'wait_for_robot', 'screw', 'release', 
            'solder', 'place', 'retreat']


ACTIONS = ['approach', 'grab_screwdriver', 'plug_screwdriver', 'grab_valve', 
           'screw_screwdriver', 'release_valve', 'grab_soldering_iron', 'plug_soldering_iron', 
           'retreat', 'join_screwdriver', 'grab_valve_terminal', 'plug_valve_terminal', 
           'place_screwdriver', 'grab_box_w_screws', 'place_box_w_screws', 'grab_hose', 
           'wait_for_robot', 'plug_hose', 'grab_box_w_membrane', 'grab_soldering_station', 
           'solder_hose', 'release_soldering_station', 'release_box_w_membrane']

MISSING_HAND_COUNT=0
NULL_COUNT=0
TOTAL_FRAME_COUNT=0
EMPTY_COUNT=0
NO_EDGE_COUNT=0

# CERTAINITY_THRESHOLD = 0.10 # out of 1


def crawl_dataset_FULL():
    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        for task in ['task_1', 'task_2', 'task_3']:
            for take in ['take_{}'.format(take_i) for take_i in range(10)]:
                yield subject, task, take

def crawl_dataset_DEBUG():
    for subject in ['subject_1', 'subject_2']:
        for task in ['task_1']:
            for take in ['take_1', 'take_7']:
                yield subject, task, take


DEBUG_MODE = False

if DEBUG_MODE:
    print("Debug mode is on\n"*5)
    crawl_dataset = crawl_dataset_DEBUG
else:
    crawl_dataset = crawl_dataset_FULL


def _get_actions_list(gt_action_json: os.PathLike, act_len: int) -> np.ndarray:
    # Returns the action of left hand and right hand, respectively.

    with open(gt_action_json) as f:
        actions_json = json.load(f)

    actions_list=np.zeros((act_len, 1), dtype=np.int32)

    # right
    for t in range(0, len(actions_json["right_hand"]) - 1, 2):

        begin = actions_json["right_hand"][t]
        right_act = actions_json["right_hand"][t + 1]
        end = actions_json["right_hand"][t + 2]

        # right_act = -1 if right_act is None else right_act

        act, obj = right_act
        if obj is None:
            right_act_str = _ACTIONS[act]
        else:
            right_act_str = _ACTIONS[act]+f"_{OBJECTS[obj]}"


        actions_list[begin:end, 0] = ACTIONS.index(right_act_str)
    
    return actions_list


def __is_list_valid(nums):
    for num in nums:
        if np.isnan(num) or np.isinf(num):
            return False
    return True


def _get_norm_func(root: str, test_subject, validation_take) -> Callable:
    # returns the normalization function for the position

    min_x, min_y, min_z = np.inf, np.inf, np.inf
    max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

    # first moment
    first_m_x, first_m_y, first_m_z = 0, 0, 0
    # second moment
    second_m_x, second_m_y, second_m_z = 0, 0, 0
    counter = 0

    for subject, task, take in crawl_dataset():     # among all the videos
        if subject == test_subject or take == validation_take:
            continue

        object_dir = opj(root, "raw", "derived_data", subject, task, take, "3d_objects")
        object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))
        for object_json in object_json_list:        # among all the frames in a video
            with open(object_json) as f:
                curr_objects = json.load(f)
            for obj in curr_objects:                # among all the objects in a frame
                bbox = obj["bounding_box"]
                # bbox: {x0, y0, z0, x1, y1, z1}
                if not __is_list_valid(bbox.values()):
                    print("ERROR "*10)
                    print("Invalid bbox: ", subject, task, take, object_json)
                    continue

                cx = (bbox["x1"] + bbox["x0"]) / 2
                cy = (bbox["y1"] + bbox["y0"]) / 2
                cz = (bbox["z1"] + bbox["z0"]) / 2
                
                min_x, min_y, min_z = min(min_x, cx), min(min_y, cy), min(min_z, cz)
                max_x, max_y, max_z = max(max_x, cx), max(max_y, cy), max(max_z, cz)

                counter += 1
                first_m_x += cx
                first_m_y += cy
                first_m_z += cz

                second_m_x += cx**2
                second_m_y += cy**2
                second_m_z += cz**2
    
    mean_x, mean_y, mean_z = first_m_x / counter, first_m_y / counter, first_m_z / counter
    std_x = np.sqrt(second_m_x / counter - mean_x**2)
    std_y = np.sqrt(second_m_y / counter - mean_y**2)
    std_z = np.sqrt(second_m_z / counter - mean_z**2)

    mean=torch.tensor([mean_x, mean_y, mean_z]).reshape(1, 3)
    std=torch.tensor([std_x, std_y, std_z]).reshape(1, 3)
    
    min_pos = torch.tensor([min_x, min_y, min_z])
    max_pos = torch.tensor([max_x, max_y, max_z])

    def normalize(pos):
        assert isinstance(pos, torch.Tensor) and pos.shape[1] == 3
        min_depth = -4000
        pos[:, 2] = torch.clip(pos[:, 2], min_depth, max_pos[2]) # clip out the outliers
        min_pos[2] = min_depth
        return (pos - min_pos) / (max_pos - min_pos)
        # return (pos - mean) / std

    return normalize

def _distance_btw_bb(bbox_a, bbox_b):
    # return np.sqrt((bbox_a[0]-bbox_b[0])**2 + (bbox_a[1]-bbox_b[1])**2 + (bbox_a[2]-bbox_b[2])**2)
    cax = (bbox_a[1] + bbox_a[0]) / 2
    cay = (bbox_a[3] + bbox_a[2]) / 2
    caz = (bbox_a[5] + bbox_a[4]) / 2
    
    cbx = (bbox_b[1] + bbox_b[0]) / 2
    cby = (bbox_b[3] + bbox_b[2]) / 2
    cbz = (bbox_b[5] + bbox_b[4]) / 2

    radicand = (cax - cbx)**2 + (cay - cby)**2 + (caz - cbz)**2
    distance = np.sqrt(radicand)

    return distance

def _is_contacting(bbox_a, bbox_b): 
    return bbox_a[0] <= bbox_b[1] and bbox_a[1] >= bbox_b[0] and bbox_a[2] <= bbox_b[3] and bbox_a[3] >= bbox_b[2] and bbox_a[4] <= bbox_b[5] and bbox_a[5] >= bbox_b[4]


def _get_relation_list(curr_objects):
    # bbox_list: N x 6 : [x0, x1, y0, y1, z0, z1]
    distance_equality_threshold = 30 # might require tuning


    bbox_list = [[obj["bounding_box"]["x0"], obj["bounding_box"]["x1"], obj["bounding_box"]["y0"], obj["bounding_box"]["y1"], obj["bounding_box"]["z0"], obj["bounding_box"]["z1"]] for obj in curr_objects]
    
    # past_bbox_list
    if curr_objects[0]["past_bounding_box"] is not None:
        past_bbox_list = [] 
        for idx, obj in enumerate(curr_objects):
            if obj["past_bounding_box"] is not None:
                past_bbox_list.append([obj["past_bounding_box"]["x0"], obj["past_bounding_box"]["x1"], obj["past_bounding_box"]["y0"], obj["past_bounding_box"]["y1"], obj["past_bounding_box"]["z0"], obj["past_bounding_box"]["z1"]])
            else:
                past_bbox_list.append(copy.deepcopy(bbox_list[idx]))
    else:
        past_bbox_list = copy.deepcopy(bbox_list)
    

    def _get_dict(oi, si, rel_name):
        return {"object_index": oi, "relation_name": rel_name, "subject_index": si}
    
    # Assumes that the objects are ordered in the same way in both bbox_list and past_bbox_list and they are the same objects. 
    # If an object is not visible, the corresponding row in bbox_list/past_bbox_list should be -1

    N=len(bbox_list)
    # relation_matrix = np.zeros((N, N, len(RELATIONS)))
    # distance_matrix = np.zeros((N, N))
    relations_dict = []
    
        
    for subject_idx in range(N):        
        for object_idx in range(subject_idx+1, N):
            
            subject_bb = bbox_list[subject_idx]
            object_bb = bbox_list[object_idx]

            # TODO: check if the -1 is the correct way to handle this
            if np.allclose(subject_bb, -1) or np.allclose(object_bb, -1):
                continue

            # # # # STATIC RELATIONS # # # #

            # contact:
            if _is_contacting(subject_bb, object_bb):
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('contact')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('contact')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'contact'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'contact'))
            
            # left of and right of:
            if subject_bb[1] < object_bb[0]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('right of')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('left of')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'right of'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'left of'))

            elif subject_bb[0] > object_bb[1]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('left of')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('right of')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'left of'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'right of'))

            # above and below:
            if subject_bb[3] < object_bb[2]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('above')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('below')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'above'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'below'))

            elif subject_bb[2] > object_bb[3]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('below')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('above')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'below'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'above'))
            
            # behind of and in front of:
            if subject_bb[5] < object_bb[4]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('in front of')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('behind of')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'in front of'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'behind of'))
            
            elif subject_bb[4] > object_bb[5]:
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('behind of')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('in front of')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'behind of'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'in front of'))
            
            # inside and surround:
            if subject_bb[0] >= object_bb[0] and subject_bb[1] <= object_bb[1] and \
               subject_bb[2] >= object_bb[2] and subject_bb[3] <= object_bb[3] and \
               subject_bb[4] >= object_bb[4] and subject_bb[5] <= object_bb[5]:
        
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('inside')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('surround')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'inside'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'surround'))
            
            elif object_bb[0] >= subject_bb[0] and object_bb[1] <= subject_bb[1] and \
                 object_bb[2] >= subject_bb[2] and object_bb[3] <= subject_bb[3] and \
                 object_bb[4] >= subject_bb[4] and object_bb[5] <= subject_bb[5]:
                
                # relation_matrix[subject_idx, object_idx, RELATIONS.index('surround')] = 1
                # relation_matrix[object_idx, subject_idx, RELATIONS.index('inside')] = 1
                relations_dict.append(_get_dict(subject_idx, object_idx, 'surround'))
                relations_dict.append(_get_dict(object_idx, subject_idx, 'inside'))


            # # # # DYNAMIC RELATIONS # # # #
            subject_past_bb = past_bbox_list[subject_idx]
            object_past_bb = past_bbox_list[object_idx]
            
            # TODO: check if the -1 is the correct way to handle this
            if np.allclose(subject_past_bb, -1) or np.allclose(object_past_bb, -1):
                continue

            delta_so=_distance_btw_bb(subject_bb, object_bb)
            delta_so_past=_distance_btw_bb(subject_past_bb, object_past_bb)

            # distance_matrix[subject_idx, object_idx] = delta_so
            # distance_matrix[object_idx, subject_idx] = delta_so

            # Şimdi de geçmişte de temas ediyorlar
            p1 = _is_contacting(subject_bb, object_bb) and _is_contacting(subject_past_bb, object_past_bb)

            # Şimdi de geçmişte de temas etmiyorlar
            p2 = not _is_contacting(subject_bb, object_bb) and not _is_contacting(subject_past_bb, object_past_bb)

            if p1:

                p3 = _distance_btw_bb(object_bb, object_past_bb) < (distance_equality_threshold / 2)
                p4 = _distance_btw_bb(subject_bb, subject_past_bb) < (distance_equality_threshold / 2)

                if p3 and p4:
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('moving together')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('moving together')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'moving together'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'moving together'))
                
                elif (not p3) and (not p4):
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('halting together')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('halting together')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'halting together'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'halting together'))
                
                elif p3 ^ p4:
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('fixed moving together')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('fixed moving together')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'fixed moving together'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'fixed moving together'))
            
            elif p2:
                if delta_so - delta_so_past < - distance_equality_threshold:
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('getting close')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('getting close')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'getting close'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'getting close'))
                
                elif delta_so - delta_so_past > distance_equality_threshold:
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('moving apart')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('moving apart')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'moving apart'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'moving apart'))
                
                else:
                    # relation_matrix[subject_idx, object_idx, RELATIONS.index('stable')] = 1
                    # relation_matrix[object_idx, subject_idx, RELATIONS.index('stable')] = 1
                    relations_dict.append(_get_dict(subject_idx, object_idx, 'stable'))
                    relations_dict.append(_get_dict(object_idx, subject_idx, 'stable'))

    return relations_dict


def create_relation_files(root: str):
    assert root == "./coax"
    raw_dir = opj(root, "raw")
    derived_dir = opj(raw_dir, "derived_data")
    for subject, task, take in crawl_dataset_FULL():
        print(".", end="", flush=True)

        relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
        object_dir = opj(derived_dir, subject, task, take, "3d_objects")
        object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))
        os.makedirs(relation_dir, exist_ok=True)
        for idx, object_json in enumerate(object_json_list):
            with open(object_json) as f:
                curr_objects = json.load(f)
            
            relation_list = _get_relation_list(curr_objects)
            relation_json_path = opj(relation_dir, f"frame_{idx:05d}.json")
            with open(relation_json_path, "w") as f:
                json.dump(relation_list, f, indent=4)

    print()

FILTERED_OBJECTS_COUNT = 0
ALL_OBJECTS_COUNT = 0
def filter_distant_objects(curr_objects, curr_relations, mask_to_delete_objs):
    global FILTERED_OBJECTS_COUNT, ALL_OBJECTS_COUNT
                     
    black_list = []
    init_len = len(curr_objects)

    for idx, obj in enumerate(curr_objects):
        if obj["bounding_box"]["z1"] < -3000 and obj["class_name"] not in ["LeftHand", "RightHand"]: # if it is too far away from the camera it is noise
            black_list.append(idx)
        
        elif obj["instance_name"] in mask_to_delete_objs: # objects that only appeared in 2% of the video
            black_list.append(idx)


    FILTERED_OBJECTS_COUNT += len(black_list)
    ALL_OBJECTS_COUNT += init_len


    if len(black_list) == 0:
        return curr_objects, curr_relations

    # Update the objects list
    curr_objects = [obj for idx, obj in enumerate(curr_objects) if idx not in black_list]

    # Update the relations list
    index_map = [-2]*init_len
    curr_idx = 0
    for idx in range(init_len):
        if idx in black_list:
            index_map[idx] = -1
        else:
            index_map[idx] = curr_idx
            curr_idx += 1

    assert len(curr_objects) == curr_idx
    assert -2 not in index_map  


    rel_idx = 0
    for _ in range(len(curr_relations)):

        new_obj_idx = index_map[curr_relations[rel_idx]["object_index"]]
        new_sub_idx = index_map[curr_relations[rel_idx]["subject_index"]]

        if new_obj_idx == -1 or new_sub_idx == -1:
            curr_relations.pop(rel_idx)
        else:
            curr_relations[rel_idx]["object_index"] = new_obj_idx
            curr_relations[rel_idx]["subject_index"] = new_sub_idx
            rel_idx += 1
    
    return curr_objects, curr_relations

def _get_graph(relation_json, object_json, vf_path, gt_actions, frame, subject, task, take, pos_normalizer, mask_to_delete_objs=None) -> gData:
    # returns the graph and its mirrored version for the given relation and object json files
    assert vf_path is None

    with open(relation_json) as f:
        curr_relations = json.load(f)

    with open(object_json) as f:
        curr_objects = json.load(f)

    if mask_to_delete_objs is not None:
        curr_objects, curr_relations = filter_distant_objects(curr_objects, curr_relations, mask_to_delete_objs)
    
    obj_names = [obj["class_name"] for obj in curr_objects]

    # # # # CHECK VALIDITY FIRST # # # #
    if obj_names.count("RightHand") != 1:
        return None
    
    if len(curr_relations) == 0 or len(curr_objects) == 0:
        return None
    # # # # CHECK VALIDITY FIRST # # # #

    
    if vf_path is not None:
        vis_feat=torch.load(vf_path)


    if len(curr_relations) == 0 or len(curr_objects) == 0:
        # Normally, invalid graphs are handled afterwards, 
        # but this is a special case, since pos depends on non-empty graphs
        print("Empty: ", subject, task, take, "f", frame)
        return None


    # __filter_low_certainty_objects(curr_objects, curr_relations)
    # curr_objects, curr_relations, filter_flag =__filter_low_certainty_objects(copy.deepcopy(curr_objects_), copy.deepcopy(curr_relations_))
    # if filter_flag:
    #     assert curr_objects == curr_objects_ and curr_relations == curr_relations_


    # initialize "Data" params
    if vf_path is None:
        x_dim = len(OBJECTS)
    else:
        x_dim = vis_feat.shape[1]

    x = torch.zeros((len(curr_objects), x_dim), dtype=torch.float)
    
    y = torch.zeros((1, len(ACTIONS)), dtype=torch.float)

    pos = torch.zeros((len(curr_objects), 3), dtype=torch.float)

    # To debug add embed some info
    # y = torch.zeros((2, len(ACTIONS)+1), dtype=torch.float)
    # y_mirrored = torch.zeros((2, len(ACTIONS)+1), dtype=torch.float)

    edge_index = []

    num_possible_relations = len(curr_objects)*(len(curr_objects)-1)

    edge_attr = torch.zeros((int(num_possible_relations), len(RELATIONS)))

    ### Fill them
    right_hand_index = -1

    for i, obj in enumerate(curr_objects):
        if vf_path is None: 
            x[i, OBJECTS.index(obj["class_name"])] = 1
        else:
            x[i] = vis_feat[i]

        pos[i, 0] = (obj["bounding_box"]["x1"] + obj["bounding_box"]["x0"]) / 2
        pos[i, 1] = (obj["bounding_box"]["y1"] + obj["bounding_box"]["y0"]) / 2
        pos[i, 2] = (obj["bounding_box"]["z1"] + obj["bounding_box"]["z0"]) / 2

        if not __is_list_valid(obj["bounding_box"].values()):
            return None

            
    # Normalize
    pos = pos_normalizer(pos)


    for relation in curr_relations:
        # edge_index must have unique edges

        oi = relation["object_index"]
        si = relation["subject_index"]

        edge=[oi, si]

        if edge not in edge_index:
            edge_index.append(edge)

        edge_attr[edge_index.index(edge), RELATIONS.index(relation["relation_name"])] = 1
        


    assert len(edge_index) <= num_possible_relations

    edge_attr=edge_attr[:len(edge_index)]
    edge_index=torch.tensor(edge_index, dtype=torch.long)


    right_act = gt_actions

    # subject, task, take were just added for debugging purposes, and will be removed
    # y[0]=torch.tensor([left_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.float)
    # y[1]=torch.tensor([right_act, int(subject[-1]+task[5]+take[-1]+str(frame))], dtype=torch.float)

    # graph_str=subject[-1]+task[5]+take[-1]+str(frame)
        
        # To debug
        # y[0, -1]=int("1"+graph_str)
        # y_mirrored[1, -1]=int("2"+graph_str)


    if right_act != -1:
        y[0, right_act] = 1

        # To debug
        # y[1, -1] = int("1"+graph_str)
        # y_mirrored[0, -1] = int("2"+graph_str)
    
    dists = []
    for edge in edge_index:
        dists.append(pos[edge[0]] - pos[edge[1]])
    dists = torch.stack(dists)


    data = gData(x=x,
                 edge_index=edge_index.t().contiguous(),
                 edge_attr=edge_attr,
                 y=y,
                 pos=pos,
                 dists=dists)

    return data

def _impute_with_previous(graph: gData, previous_graph: gData, validity: str) -> gData:
    # For now, just returns the previous graph
    # If required, impute using the data of previous graph
    return previous_graph.clone()
    

def download_coax_dataset(root: str, download_anyway) -> None:
    # Downloads and unzip the Coax dataset to raw_dir.
    # Check if zips are already downloaded
    raw_dir = opj(root, "raw")
    is_something_missing = False
    for zipf in ["action_ground_truth.zip", 
                 "derived_data.zip"]:
        if not os.path.exists(opj(raw_dir, zipf)):
            print(f"{zipf} is missing, downloading all...")
            is_something_missing = True
            # if single file is missing, download all bc it could be corrupted
            break

    urls = [
        "https://www2.hs-esslingen.de/public/Faculty-IT-Media/_public/coax-dataset/action_ground_truth.zip",
        "https://www2.hs-esslingen.de/public/Faculty-IT-Media/_public/coax-dataset/derived_data.zip",        
    ]
    # Download
    if download_anyway or is_something_missing:
        for url in urls:
            path = download_url(url, raw_dir)

        
    # remove everything except the zips (otherwise it might be corrupted)
    for f in os.listdir(raw_dir):
        if f.endswith(".zip"):
            continue
        else:
            print(f"Removing old files: {f}")
            shutil.rmtree(opj(raw_dir, f))

    # Unzip
    for url in urls:
        path = opj(raw_dir, url.split("/")[-1])
        extract_zip(path, raw_dir)
    

def _get_max_length(root: str, downsample) -> int:
    # Returns the maximum length of the videos (used for padding)
    max_length = 0
    for subject, task, take in crawl_dataset():
        relation_dir = opj(root, "raw", "derived_data", subject, task, take, "spatial_relations")
        curr_list = glob.glob(opj(relation_dir, "*.json"))[::downsample]
        curr_length = len(curr_list)
        max_length = max(max_length, curr_length)
    return max_length


def __get_padding_graph() -> gData:
    # Returns a dummy graph for padding
    # To separate dummy from real check y.sum() == -1

    # To make hand_pooling keep working, this should have left and right hand 
    x = torch.zeros((2, len(OBJECTS)), dtype=torch.float)
    x[0, OBJECTS.index("LeftHand")] = 1
    x[1, OBJECTS.index("RightHand")] = 1
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, len(RELATIONS)), dtype=torch.float)
    edge_attr[0, RELATIONS.index("contact")] = 1
    edge_attr[1, RELATIONS.index("contact")] = 1
    
    y = torch.zeros((2, len(ACTIONS)), dtype=torch.float)
    y[0, 0] = -0.5
    y[1, 0] = -0.5
    return gData(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)

def _pad_sequence(data_list: List[gData], target_length: int) -> List[gData]:
    # Pads the sequence to target_length
    # If the sequence is already longer than target_length, it is not padded
    # If the sequence is shorter than target_length, it is padded with a dummy graph
    if len(data_list) < target_length:
        for _ in range(target_length - len(data_list)):
            data_list.append(__get_padding_graph())


def get_processing_dir(root: str, temporal_length: int, downsample: int, use_vf: bool, filtered_data: bool, monograph: bool) -> str:
    # Naming convention for the processed directory

    f_name = "processed"
    f_name += "W" if temporal_length == -1 else str(temporal_length)
    f_name += f"_d{downsample}" if downsample > 1 else ""
    f_name += "_vf" if use_vf else ""
    f_name += "_G" if monograph else ""
    f_name += "_F" if filtered_data else ""

    return opj(root, f_name)

def get_mask_to_delete_objs(object_json_list):
    threshold = 0.02 # if the object is not visible in 2% of the frames, delete it

    mask = [] # list of instance_name s

    obj_dict = {}

    for t, object_path in enumerate(object_json_list):
        with open(object_path) as f:
            objects_3d = json.load(f)

        # objects_3d: list of dictionaries

        instance_names = [obj["instance_name"] for obj in objects_3d]

        for obj in set(instance_names).union(set(obj_dict.keys())):
            
            if obj in instance_names and obj in obj_dict:
                obj_dict[obj].append(1)

            elif obj in obj_dict and obj not in instance_names:
                obj_dict[obj].append(0)
            
            elif obj not in obj_dict and obj in instance_names:
                obj_dict[obj] = [0]*t
                obj_dict[obj].append(1)

    # appearance_ratio = {obj: sum(mask_list) / len(mask_list) for obj, mask_list in obj_dict.items()}

    for obj, mask_list in obj_dict.items():
        if sum(mask_list) / len(mask_list) < threshold:
            mask.append(obj)

    return mask

def process_coax_dataset(root: str, temporal_length: int, downsample: int, use_vis_feat: bool, filtered_data: bool) -> None:
    assert not use_vis_feat
    assert root == "./coax"
    global TOTAL_FRAME_COUNT
    # temporal_length defines the number of frames in a sample regardless of the downsampling rate

    processed_dir = get_processing_dir(root, temporal_length, downsample, use_vis_feat, filtered_data, monograph=False)
    os.mkdir(processed_dir)
    
    if temporal_length == -1:
        process_whole_video = True
    else:
        process_whole_video = False


    for subject in ['subject_{}'.format(i) for i in range(1, 7)]:
        os.mkdir(opj(processed_dir, subject))


    raw_dir = opj(root, "raw")
    derived_dir = opj(raw_dir, "derived_data")
    gt_dir = opj(raw_dir, "action_ground_truth")
    
    # if use_vis_feat:
    #     vf_master_dir = opj(raw_dir, "vis_feat")
    #     assert os.path.exists(vf_master_dir), f"Vis feat is not available in {vf_master_dir}"

    sample_idx=0

    imputation_count=0
    prepended_things_count=0

    dataset_mapping_master = {"sample":{}, "vid":{}}

    # To double check
    sample_count = 0

    # Progress bar
    progressbar_manager=enlighten.get_manager()
    ticks=progressbar_manager.counter(total=180, desc="Progress:", unit="folder", color="blue")

    for subject, task, take in crawl_dataset():

        subject_int, task_int, take_int = int(subject[-1]), int(task[5]), int(take[-1])
        
        if take_int == 0 and task_int == 1: # For every subject re-create the normalizer (task idx starts from 1)
            pos_normalizer = _get_norm_func(root, subject, take)
        # Sometimes initial graphs are invalid, so imputation is not working. This is to fix that
        prepend_count=0

        # make sure there is no cross-folder graph sequence
        data_list = []

        object_dir = opj(derived_dir, subject, task, take, "3d_objects")
        object_json_list = natsort.natsorted(glob.glob(opj(object_dir, "*.json")))

        relation_dir = opj(derived_dir, subject, task, take, "spatial_relations")
        relation_json_list = natsort.natsorted(glob.glob(opj(relation_dir, "*.json")))

        # if use_vis_feat:
        #     vf_dir = opj(vf_master_dir, subject, task, take)
        #     vf_list = natsort.natsorted(glob.glob(opj(vf_dir, "early_feat_*.pt")))


        gt_action_json = opj(gt_dir, subject, task, f"{take}.json")
        act_list_full=_get_actions_list(gt_action_json, len(relation_json_list))

        assert int(pathlib.Path(relation_json_list[0]).stem.split("_")[1]) == 0, f"{relation_json_list[0]=}"
        assert int(pathlib.Path(relation_json_list[-1]).stem.split("_")[1]) == len(relation_json_list) - 1, f"{relation_json_list[-1]=}, {len(relation_json_list)=}"
        
        assert int(pathlib.Path(object_json_list[0]).stem.split("_")[1]) == 0, f"{object_json_list[0]=}"
        assert int(pathlib.Path(object_json_list[-1]).stem.split("_")[1]) == len(object_json_list) - 1, f"{object_json_list[-1]=}, {len(object_json_list)=}"

        # Downsample
        relation_json_list = relation_json_list[::downsample]
        object_json_list = object_json_list[::downsample]
        act_list = act_list_full[::downsample]

        assert len(relation_json_list) == len(object_json_list)
        assert len(relation_json_list) == len(act_list)
        assert len(relation_json_list) > 0

        # if use_vis_feat:
        #     vf_list = vf_list[::downsample]
        #     assert len(relation_json_list) == len(vf_list), f"{subject=}, {task=}, {take=}, {len(relation_json_list)=}, {len(vf_list)=}"

        # Just to make sure at least one sample is saved per video. 
        # Mostly used for debugging whole video processing
        successful_save = False


        dataset_mapping_master["vid"][str(subject_int)+str(task_int)+str(take_int)] = act_list_full
        sample_count = sample_count + len(act_list) - temporal_length + 1 # This is redundant for whole video processing   

        # if use_vis_feat:
        #     things_to_zip = zip(relation_json_list, object_json_list, vf_list)
        # else:
        things_to_zip = zip(relation_json_list, object_json_list)

        if process_whole_video:
            temporal_length = len(relation_json_list)

        if filtered_data:
            mask_to_delete_objs = get_mask_to_delete_objs(object_json_list) 
        else:
            mask_to_delete_objs = None

        for frame, paths in enumerate(things_to_zip):
            
            if use_vis_feat:
                relation_json, object_json, vf_path = paths
            else:
                relation_json, object_json = paths
                vf_path = None

            TOTAL_FRAME_COUNT += 1

            graph = _get_graph(relation_json, object_json, vf_path, act_list[frame], frame, subject, task, take, pos_normalizer, mask_to_delete_objs)

            ### Prepending and imputation ###
            # If the graph is not valid, try to fix it (impute maybe). 
            # If it is not possible resets the sequence and continues
            # validity = _check_validity(graph)
            validity = "valid" if graph is not None else "not valid"
            
            if validity == "valid":
                pass
            # elif validity in ["null"]:
            #     raise Exception(f"Unhandled graph error: {validity}")
            elif len(data_list) > 0:
                # use previous graph to impute
                graph = _impute_with_previous(graph, data_list[-1], validity)
                imputation_count+=1
            else:
                # print("Skipping: invalid graph and no previous graph to impute: ", subject, task, take, "f", frame)
                prepend_count+=1
                prepended_things_count+=1
                # data_list=[]
                # data_list_mirrored=[]
                continue
            

            data_list.append(graph)    
            

            if prepend_count != 0 and len(data_list) == 1:
                for _ in range(prepend_count):
                    data_list.append(graph.clone())
                assert len(data_list) == prepend_count+1
                prepend_count=0
            ##################################

                
            # if the sequence is complete, saves it
            if len(data_list) == temporal_length:
    
                torch.save(data_list, opj(processed_dir, subject, f'data_{sample_idx}.pt'))
                dataset_mapping_master["sample"][sample_idx] = (subject_int, task_int, take_int, frame-temporal_length + 1, frame + 1)
                sample_idx+=1

                successful_save = True
                data_list[0].validate(raise_on_error=True)
                data_list.pop(0)
        # End of for loop (frame)
                
        if not successful_save:
            raise ValueError("WTF")

        if not process_whole_video:
            assert sample_count == sample_idx, f"{sample_count=}, {sample_idx=}"

        ticks.update()

    # End of for loop (video)
    progressbar_manager.stop()

    print(f"\n\n#######\n{TOTAL_FRAME_COUNT=}, {EMPTY_COUNT=}, {NO_EDGE_COUNT=}, {MISSING_HAND_COUNT=}, {NULL_COUNT=}")
    
    print(f"{imputation_count=}, {prepended_things_count=}")
    if ALL_OBJECTS_COUNT != 0:
        print(f"Ratio of object filtering: {FILTERED_OBJECTS_COUNT/ALL_OBJECTS_COUNT:.2f}")

    config_info = {"sample_count": sample_idx, 
                   "temporal_length": temporal_length,
                   "downsample": downsample,
                   "filter": filtered_data,
                   "git_hash": subprocess.check_output(["git", "describe", "--always"]).strip().decode()}

    with open(opj(processed_dir, "config_info_master.json"), "w") as f:
        json.dump(config_info, f)
        
    with open(opj(processed_dir, "dataset_mapping_master.pickle"), "wb") as f:
        pickle.dump(dataset_mapping_master, f)


    if process_whole_video and DEBUG_MODE is False:
        assert sample_idx == 1080, f"{sample_idx=} != 1080"
    else:
        assert sample_idx == sample_count, f"{sample_idx=} ,{sample_count=}"


class CoaxDataset(gDataset):
    def __init__(self, root, temporal_length, downsample, mapping, use_vf, filtered_data, monograph):
        assert not use_vf
        assert root == "./coax"
        self.temporal_length = temporal_length

        self.pdir = get_processing_dir(root, temporal_length, downsample, use_vf, filtered_data, monograph)
        
        self.mapping = mapping

        self.split_len = len(mapping)
        
        self.distribution = torch.zeros((len(ACTIONS)))

        transform, pre_transform, pre_filter = None, None, None
        super().__init__(root, transform, pre_transform, pre_filter)

    # This is weird
    @property
    def processed_dir(self) -> str:
        return self.pdir

    @property
    def raw_file_names(self):
        return ["action_ground_truth/subject_1/task_1/take_0.json"]

    @property
    def processed_file_names(self):
        addr = [opj("subject_1", 'data_0.pt')] # sample data
        return addr


    def len(self) -> int:
        return self.split_len

    def get(self, idx: int) -> gData:

        path, global_idx = self.mapping[idx]
        data = torch.load(opj(self.processed_dir, path))

        return global_idx, data



def create_mappings(processed_dir: str, temporal_length: int, test_subject: int, validation_take: int) -> Tuple[dict, dict]:
    
    path_mapping = {"train": {}, "test": {}, "val": {}} # [split]: local idx -> path, global idx (i.e. idx = local; train_idx,val_idx,test_idx = global)

    dataset_mapping = {"train":{"vid":{}, "sample":{}},
                       "test":{"vid":{}, "sample":{}},
                       "val":{"vid":{}, "sample":{}}}
    
    with open(opj(processed_dir, "config_info_master.json"), "r") as f:
        config_master = json.load(f)


    with open(opj(processed_dir, "dataset_mapping_master.pickle"), "rb") as f:
        dataset_mapping_master = pickle.load(f)


    # vids
    # dataset_mapping_master["vid"][str(subject_int)+str(task_int)+str(take_int)] = act_list
    for id, act_list in dataset_mapping_master["vid"].items():

        subject_int, task_int, take_int = int(id[0]), int(id[1]), int(id[2])

        if subject_int == test_subject:
            dataset_mapping["test"]["vid"][id] = act_list
        elif take_int == validation_take:
            dataset_mapping["val"]["vid"][id] = act_list
        else:
            dataset_mapping["train"]["vid"][id] = act_list


    # samples
    test_idx = 0
    val_idx = 0
    train_idx = 0

    # dataset_mapping_master["sample"][sample_idx] = (subject_int, task_int, take_int, frame-temporal_length + 1, frame + 1, "_" or "M") 
    for idx, (sub, task, take, start, end) in dataset_mapping_master["sample"].items():
        
        if sub == test_subject:
            dataset_mapping["test"]["sample"][idx] = (sub, task, take, start, end)
            path_mapping["test"][test_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            test_idx+=1
        
        elif take == validation_take:
            dataset_mapping["val"]["sample"][idx] = (sub, task, take, start, end)
            path_mapping["val"][val_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            val_idx+=1

        else:
            dataset_mapping["train"]["sample"][idx] = (sub, task, take, start, end)
            path_mapping["train"][train_idx] = (opj("subject_"+str(sub), f"data_{idx}.pt"), idx)
            train_idx+=1

    
    assert train_idx + test_idx + val_idx == config_master["sample_count"], f"{train_idx + test_idx + val_idx=}, {config_master['sample_count']=}" 


    return dataset_mapping, path_mapping


# Simple test
def print_data_point(data: gData):

    print("Summary:")
    print(f"{data.num_nodes=}")
    print(f"{data.num_edges=}")
    print(f"{data.num_node_features=}")
    print(f"{data.is_directed()=}")

    print("validate: ", data.validate(raise_on_error=True))

    print("\nObjects:")
    for i in torch.argmax(data.x, dim=1):
        print(OBJECTS[i])

    assert data.edge_index.shape[1] == data.edge_attr.shape[0]

    print("\nEdges:", end="")
    for i in range(len(data.edge_index.t())):
        obj1, obj2 = data.edge_index.t()[i]
        print("\n\n", obj1.item(), " to ", obj2.item(), ": ", end="")

        for j in torch.nonzero(data.edge_attr[i]):
            print(RELATIONS[j], end=", ")

    print("\n\nActions:")
    print("Left:", ACTIONS[torch.argmax(data.y[0, :-1], dim=0)], ", Right: ", ACTIONS[torch.argmax(data.y[1, :-1], dim=0)])

    print("Graph id: ", data.y[0,-1].item())


def get_coax_dataset_splits(*, root: str, process_it_anyway: bool, test_subject: int, validation_take: int , temporal_length: int, downsample: int, use_vf: bool, filtered_data:bool, monograph: bool) -> Tuple[gDataset, gDataset, gDataset]:
    # dataset_mapping = {"train":{"vid":{}, "sample":{}},
    #                    "test":{"vid":{}, "sample":{}},
    #                    "val":{"vid":{}, "sample":{}}}
    
    # path_mapping = {"train": {}, "test": {}, "val": {}} # [split]: idx -> path, global_idx

    processed_dir = get_processing_dir(root, temporal_length, downsample, use_vf, filtered_data, monograph)

    
    assert os.path.isfile(opj(processed_dir, "dataset_mapping_master.pickle")), f"In {processed_dir}, dataset is not preprocessed yet"
    assert os.path.isfile(opj(processed_dir, "config_info_master.json")), f"In {processed_dir}, dataset is not preprocessed yet"

    if os.path.exists(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle")) and (not process_it_anyway):
        # read from folder
        print("Reading dataset mappings...")
        with open(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle"), "rb") as f:
            dataset_mapping = pickle.load(f)
        
        with open(opj(processed_dir, f"path_mapping_fold{test_subject}.pickle"), "rb") as f:
            path_mapping = pickle.load(f)

    else:
        print("Creating dataset mappings...")

        dataset_mapping, path_mapping = create_mappings(processed_dir, temporal_length, test_subject, validation_take)

        with open(opj(processed_dir, f"dataset_mapping_fold{test_subject}.pickle"), "wb") as f:
            pickle.dump(dataset_mapping, f)

        with open(opj(processed_dir, f"path_mapping_fold{test_subject}.pickle"), "wb") as f:
            pickle.dump(path_mapping, f)


    # calculate class distribution

    distibution = torch.zeros((len(ACTIONS)))

    for act_seq in dataset_mapping["train"]["vid"].values():
        act, cnt = torch.unique(torch.tensor(act_seq.flatten()), return_counts=True)
        for a, c in zip(act, cnt):
            distibution[a] += c


    train_set=CoaxDataset(root, temporal_length, downsample, path_mapping["train"], use_vf, filtered_data, monograph)
    test_set=CoaxDataset(root, temporal_length, downsample, path_mapping["test"], use_vf, filtered_data, monograph)
    val_set=CoaxDataset(root, temporal_length, downsample, path_mapping["val"], use_vf, filtered_data, monograph)
    
    train_set.distribution = distibution

    return train_set, test_set, val_set

if __name__ == "__main__":

    # a, b, c=get_coax_dataset_splits(root=".", 
    #                                     process_it_anyway=False, 
    #                                     test_subject=6, 
    #                                     validation_take=7, 
    #                                     temporal_length=10, 
    #                                     downsample=3, 
    #                                     use_vf=False,
    #                                     filtered_data=True,
    #                                     monograph=False)   
    # print(len(a), len(b), len(c))
    # if not os.path.exists("raw_coax"):
    #     os.makedirs("raw_coax")
    # download_coax_dataset(root="./coax", download_anyway=True)

    # create_relation_files(raw_dir="./coax/raw")

    gt_dir = "./coax/raw/action_ground_truth"
    counter = Counter()
    for subject, task, take in crawl_dataset_FULL():
        relation_json_list = glob.glob(opj(gt_dir, subject, task, f"{take}*.json"))

        gt_action_json = opj(gt_dir, subject, task, f"{take}.json")
        act_list_full=_get_actions_list(gt_action_json, len(relation_json_list), counter)

    actions=[]
    for pairs, _ in counter.items():
        act, obj = pairs
        if obj is None:
            actions.append(_ACTIONS[act])
        else:
            actions.append(_ACTIONS[act]+f"_{OBJECTS[obj]}")

    print(counter)
    print(actions)
