# Logical Graphs Generator v2.0

from typing import Dict, Any, List, Tuple
import os
import networkx as nx
import json
import random
import numpy as np
import copy
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from dataclasses import dataclass, field
from itertools import count

from collections import Counter
from sklearn.model_selection import train_test_split

## Utility functions


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")
    print("Wrote {} records to {}".format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    print("Loaded {} records from {}".format(len(data), input_path))
    return data


def get_edges(path):
    return set([(path[pi], path[pi + 1]) for pi in range(len(path) - 1)])


def verify_and_load_rule_store(loc):
    """Given a json rule store, it must be of the form
        "r_1,r_2" -> "r_3"
        Then return and load the rule store
    Args:
        loc ([type]): location of the rule store, in .json file
    """
    rule = json.load(open(loc))
    for k, v in rule.items():
        assert type(k) == str
        assert len(k.split(",")) == 2
        assert type(v) == str
    return rule


def del_simple_paths(sub_e, aps, path):
    # print(len(aps))
    sub_e = copy.deepcopy(sub_e)
    e_to_del = set()
    path_edges = get_edges(path)
    for ap in aps:
        es = get_edges(ap)
        es = es - es.intersection(path_edges)
        if len(es) > 0:  # and len(e_to_del.intersection(es)) == 0:
            e_to_del.add(random.choice(list(es)))
    sub_e = [e for e in sub_e if e not in e_to_del]
    return sub_e


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


## Generator functions


def create_maps(rules):
    """Create dictionary mapping rule positions

    Args:
        rules ([type]): [description]

    Returns:
        [type]: [description]
    """
    body_0_map = {}
    body_1_map = {}
    head_map = {}
    for rule in rules:
        body, head = rule
        if body[0] not in body_0_map:
            body_0_map[body[0]] = []
        body_0_map[body[0]].append(rule)
        if body[1] not in body_1_map:
            body_1_map[body[1]] = []
        body_1_map[body[1]].append(rule)
        if head not in head_map:
            head_map[head] = []
        head_map[head].append(rule)
    return body_0_map, body_1_map, head_map


def sample_path_and_target(
    rule_world: Dict[str, str],
    max_path_len: int = 5,
    random_path_len: bool = True,
    debug: bool = False,
    last_node: int = 0,
) -> Tuple[List[Tuple[int, int, str]], List[str], List[int], int, int, str, List[str]]:
    """Sample a path and target for the graph
    Given a rule dict:
    - randomly sample a target relation
    - iteratively substitute the body of this relation to get a path

    Args:
        rule_world (Dict[str, str]): [description]
        max_path_len (int, optional): [description]. Defaults to 5.
        random_path_len (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        edge_list (List[Tuple[int, int, str]]): list of edges
        rules_used (List[str]): rules used in path
        rules_used_pos (List[int]): position of rules which are inserted in the rule path
        source (int): source node
        sink (int): sink node
        target (str): target relation
        sampled_rule (List[str]): list of relations in the path

    """
    ## set target
    rules_used = []
    rules_used_pos = []
    all_targets = list(set([head for body, head in rule_world.items()]))
    sample_target = random.choice(all_targets)
    sampled_rule = random.choice(
        [body for body, head in rule_world.items() if head == sample_target]
    )
    target = rule_world[sampled_rule]
    if debug:
        print(target)
    rules_used.append(sampled_rule)
    rules_used_pos.append(0)
    sampled_rule = sampled_rule.split(",")

    if random_path_len:
        path_length = random.choice(range(max_path_len - 1))
    else:
        path_length = max_path_len - 1

    ## substitution mechanism
    for i in range(path_length - 1):
        replace_pos = random.choice(range(len(sampled_rule)))
        replaced_head = sampled_rule[replace_pos]
        potential_bodies = [
            body for body, head in rule_world.items() if head == replaced_head
        ]
        if len(potential_bodies) > 0:
            cand_body = random.choice(potential_bodies)
            rules_used.append(cand_body)
            rules_used_pos.append(replace_pos)
            cand_body = cand_body.split(",")
            sampled_rule[replace_pos] = cand_body[1]
            sampled_rule.insert(replace_pos, cand_body[0])

    ## add edges
    edges = []
    last_node = last_node
    for sr in sampled_rule:
        u = last_node
        v = u + 1
        last_node += 1
        edges.append((u, v, sr))
    source = 0
    sink = last_node
    return edges, rules_used, rules_used_pos, source, sink, target, sampled_rule


def expansion_step(
    edges: List[Tuple[int, int, str]],
    body_0_map: Dict[str, Tuple[List[str], str]],
    body_1_map: Dict[str, Tuple[List[str], str]],
    head_map: Dict[str, Tuple[List[str], str]],
    new_node_id: int = 0,
    search_ct: int = 100,
) -> List[Tuple[int, int, str]]:
    """generate one expansion step which results in one new node
    and two new edges.
    First, sample an existing edge to make the expansion, (u,v,R)
    The two new edges can be added in either of 3 ways:
        - "head" if R can be substituted in R_k, then add (u,x, R_i) and (x,v, R_j)
        - "body_0" if R can be substituted in R_i, then add (v,x, R_j) and (u,x, R_k)
        - "body_1" if R can be substituted in R_j, then add (x,u, R_i) and (x,v, R_k)

    Args:
        edges (List[Tuple[int, int, str]]): [description]
        body_0_map (Dict[str, Tuple[int, int]]): [description]
        body_1_map (Dict[str, Tuple[int, int]]): [description]
        head_map (Dict[str, int]): [description]
        new_node_id (int, optional): [description]. Defaults to 0.
        search_ct (int, optional): [description]. Defaults to 100.

    Returns:
        List[Tuple[int, int, str]]: [description]
    """
    available_steps = []
    u, v, R = -1, -1, ""
    for _ in range(search_ct):
        u, v, R = random.choice(edges)
        if R in body_0_map:
            available_steps.append("body_0")
        if R in body_1_map:
            available_steps.append("body_1")
        if R in head_map:
            available_steps.append("head")
        if len(available_steps) > 0:
            break
    if len(available_steps) == 0:
        return edges
    expansion_policy = random.choice(available_steps)
    if expansion_policy == "head":
        sample_rule = random.choice(head_map[R])
        edges.append((u, new_node_id, sample_rule[0][0]))
        edges.append((new_node_id, v, sample_rule[0][1]))
    elif expansion_policy == "body_0":
        sample_rule = random.choice(body_0_map[R])
        edges.append((v, new_node_id, sample_rule[0][1]))
        edges.append((u, new_node_id, sample_rule[1]))
    elif expansion_policy == "body_1":
        sample_rule = random.choice(body_1_map[R])
        edges.append((new_node_id, u, sample_rule[0][1]))
        edges.append((new_node_id, v, sample_rule[1]))
    return edges


def completion_step(
    edges: List[Tuple[int, int, str]], rules: List[Tuple[List[str], str]]
) -> List[Tuple[int, int, str]]:
    """generate one completion step, where one new edge is added among existing nodes
    randomly sample an edge (u,v,R)
    one new edge can be added in the following ways:
        - a: substitute R for R_i, sample (v,x, R_j), add if not present (u,x, R_k)
        - b: substitute R for R_j, sample (x,u, R_i), add if not present (x,v, R_k)
        - c: substitute R for R_k, sample (u,x, R_i), add if not present (x,v, R_j)
        - d: substitute R for R_k, sample (x,v, R_j), add if not present (u,x, R_i)
    Args:
        edges ([type]): [description]
    Returns:
        [type]: [description]
    """
    edge_set = set(edges)
    edge_no_rel = set([(e[0], e[1]) for e in edges])

    u, v, R = random.choice(edges)

    u_0 = [e for e in edges if e[0] == u]
    u_1 = [e for e in edges if e[1] == u]
    v_0 = [e for e in edges if e[0] == v]
    v_1 = [e for e in edges if e[1] == v]

    if len(u_0) > 0:
        R_k = R
        _, x, R_i = random.choice(u_0)
        cand_rules = [rule for rule in rules if rule[1] == R_k and rule[0][0] == R_i]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_j = inject_rule[0][1]
            if (x, v) not in edge_no_rel:
                edge_set.add((x, v, R_j))
                edge_no_rel.add((x, v))

    if len(u_1) > 0:
        R_j = R
        x, _, R_i = random.choice(u_1)
        cand_rules = [rule for rule in rules if rule[0][1] == R_j and rule[0][0] == R_i]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_k = inject_rule[1]
            if (x, v) not in edge_no_rel:
                edge_set.add((x, v, R_k))
                edge_no_rel.add((x, v))

    if len(v_0) > 0:
        R_i = R
        _, x, R_j = random.choice(v_0)
        cand_rules = [rule for rule in rules if rule[0][0] == R_i and rule[0][1] == R_j]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_k = inject_rule[1]
            if (u, x) not in edge_no_rel:
                edge_set.add((u, x, R_k))
                edge_no_rel.add((u, x))

    if len(v_1) > 0:
        R_k = R
        x, _, R_j = random.choice(v_1)
        cand_rules = [rule for rule in rules if rule[1] == R_k and rule[0][1] == R_j]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_i = inject_rule[0][0]
            if (u, x) not in edge_no_rel:
                edge_set.add((u, x, R_i))
                edge_no_rel.add((u, x))

    edges = list(edge_set)

    return edges


@dataclass
class GraphRow:
    """
    Graph Dataset row
    """

    edges: List[Tuple[int, int, str]]
    source: int
    sink: int
    target: str
    query: List[int]
    descriptor: str
    rules_used: List[str]
    rules_used_pos: List[int]
    resolution_path: List[int]
    noise_edges: List[Tuple[int, int, str]]
    gid: int = field(default_factory=count().__next__, init=False)

    def get_str_without_noise(self):
        return json.dumps(
            {
                "edges": self.edges,
                "source": self.source,
                "sink": self.sink,
                "query": self.query,
                "target": self.target,
                "descriptor": self.descriptor,
                "rules_used": self.rules_used,
                "rules_used_pos": self.rules_used_pos,
                "resolution_path": self.resolution_path,
            }
        )

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.get_str_without_noise() == other.get_str_without_noise()

    def __hash__(self):
        return self.gid


def sample_graph(
    rule_world: Dict[str, str],
    max_path_len: int = 5,
    num_steps: int = 50,
    expansion_prob: float = 0.5,
    num_completion_steps: int = 10,
    debug: bool = False,
    random_path_len: bool = True,
) -> GraphRow:
    """Main graph generation logic

    Args:
        rule_world (Dict): dictionary of rules used to generate the graph
        max_path_len (int, optional): Path length of the descriptor.
            Higher number would lead to more difficulty. Defaults to 5.
        add_noise (bool): If true, add the noise (dangling, disconnected or supporting components) in the graph
        num_steps (int, optional): Maximum number of expansion steps used in a graph.
            Higher number results to more complex graphs. Defaults to 50.
        expansion_prob (float, optional): Probability of expansion of the graph in each step
        num_completion_steps (int, optional): Max number of steps to run graph completion.
            Higher number results to dense graphs. Defaults to 10.
        debug (bool, optional): [description]. Defaults to False.
        random_path_len (bool, optional): Defaults to True. If False, fixes randomization of graph length, and is always max_path_length + 1

    Returns a GraphRow object containing:
        edges: list of edges of the graph (u,v,r)
        source: source node id
        sink: sink node id
        target: target relation to predict
        descriptor: descriptor of the graph, which is the concatenation of the relations in the shortest path from source to sink.
        rules_used: rules used to derive to the shortest path
        path: shortest path from the source to sink
        noise_edges: list of edges of the graph (u,v,r) which are not required to solve the task
    """
    rules = [(body.split(","), head) for body, head in rule_world.items()]
    body_0_map, body_1_map, head_map = create_maps(rules)

    ## generate resolution paths
    if debug:
        print("Sampling resolution path ...")

    (
        edges,
        rules_used,
        rules_used_pos,
        source,
        sink,
        target,
        sampled_rule,
    ) = sample_path_and_target(
        rule_world=rule_world,
        max_path_len=max_path_len,
        random_path_len=random_path_len,
    )

    path_length = sink + 1
    path = list(range(path_length))
    noise_edges = copy.deepcopy(edges)
    # Noise: sample neighbors
    if debug:
        print("Sampling neighbors ...")
    new_node = sink + 1
    for _ in range(num_steps):
        if random.uniform(0, 1) > expansion_prob:
            noise_edges = expansion_step(
                noise_edges, body_0_map, body_1_map, head_map, new_node
            )
        new_node += 1
        # run completion step for n numbers
        for _ in range(random.choice(range(1, num_completion_steps))):
            noise_edges = completion_step(noise_edges, rules)
    ## remove self loops in noise
    noise_edges = [e for e in noise_edges if e[0] != e[1]]
    if debug:
        print("Saving edge to rel map ...")
    edge2rel = {(e[0], e[1]): e[2] for e in edges}
    g = nx.DiGraph()
    edges_to_add = [(e[0], e[1]) for e in edges]
    g.add_edges_from(edges_to_add)
    noise_edge2rel = {
        (e[0], e[1]): e[2] for e in noise_edges if (e[0], e[1]) not in edges_to_add
    }
    noise_g = nx.DiGraph()
    noise_edges_to_add = [(e[0], e[1]) for e in noise_edges]
    # When creating noise digraph, add all edges from the clean graph as well
    # This is needed to compute shortest paths
    noise_g.add_edges_from(noise_edges_to_add + edges_to_add)
    if debug:
        print(nx.info(g))

    if debug:
        print("Removing shortest paths ...")
    sub_e = list(g.edges)
    ## If noise has been added, Compute and remove shortest paths

    sub_eg = list(noise_g.edges)
    aps = list(nx.all_simple_paths(noise_g, source, sink, cutoff=path_length))
    # print(len(aps))
    sub_eg = del_simple_paths(sub_eg, aps, path)
    g = nx.from_edgelist(sub_eg, create_using=nx.DiGraph)
    # print('Computing shortest paths from sink to source')
    aps = list(nx.all_simple_paths(noise_g, path[-1], path[0], cutoff=path_length))
    # print(len(aps))
    sub_eg = del_simple_paths(sub_eg, aps, path)
    # sub_eg.extend(list(get_edges(path)))

    if debug:
        print("Done, computing stats of new graph")
        g = nx.DiGraph()
        g.add_edges_from([(e[0], e[1]) for e in sub_e])
        print(nx.info(g))

    if debug:
        print("Getting final edges")
    sub_e = list(set(sub_e))
    edges = [(e[0], e[1], edge2rel[(e[0], e[1])]) for e in sub_e]
    # sort edges
    edges = list(sorted(edges, key=lambda tup: tup[1]))
    # for noise
    sub_eg = list(set(sub_eg))
    noise_edges = [
        (e[0], e[1], noise_edge2rel[(e[0], e[1])])
        for e in sub_eg
        if (e[0], e[1]) in noise_edge2rel
    ]
    # sort edges
    noise_edges = list(sorted(noise_edges, key=lambda tup: tup[1]))
    descriptor = ",".join(sampled_rule)
    query = [source, sink, target]
    return GraphRow(
        edges,
        source,
        sink,
        target,
        query,
        descriptor,
        rules_used,
        rules_used_pos,
        path,
        noise_edges,
    )


def sample_world_graph(
    rule_world: Dict[str, str],
    max_path_len: int = 5,
    num_sampled_paths: int = 10,
    num_steps: int = 50,
    expansion_prob: float = 0.5,
    num_completion_steps: int = 10,
    debug: bool = False,
) -> List[Tuple[int, int, str]]:
    """Sample world graph given a rule world

    Args:
        rule_world (Dict[str, str]): [description]
        max_path_len (int, optional): [description]. Defaults to 5.
        num_steps (int, optional): [description]. Defaults to 50.
        expansion_prob (float, optional): [description]. Defaults to 0.5.
        num_completion_steps (int, optional): [description]. Defaults to 10.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        List[Tuple[int, int, str]]: [description]
    """
    rules = [(body.split(","), head) for body, head in rule_world.items()]
    body_0_map, body_1_map, head_map = create_maps(rules)
    # Sample multiple paths from the rule world
    all_edges = []
    last_node = 0
    for _ in range(num_sampled_paths):
        (edges, _, _, _, last_node, _, _) = sample_path_and_target(
            rule_world=rule_world, max_path_len=max_path_len, last_node=last_node
        )
        all_edges.extend(edges)
    # Complete the graph by repeated execution of expansion and contraction
    new_node = last_node + 1
    for _ in range(num_steps):
        if random.uniform(0, 1) > expansion_prob:
            all_edges = expansion_step(
                all_edges, body_0_map, body_1_map, head_map, new_node
            )
        new_node += 1
        # run completion step for n numbers
        for _ in range(random.choice(range(1, num_completion_steps))):
            all_edges = completion_step(all_edges, rules)
    print("Complete")
    if debug:
        print("Done, computing stats of world graph")
        g = nx.DiGraph()
        g.add_edges_from([(e[0], e[1]) for e in all_edges])
        print(nx.info(g))

    return all_edges


## Splitting logic


def get_des_ids(sp):
    if len(sp) > 0:
        return [int(x) for x in sp.split(",")]
    else:
        return []


def re_index_nodes(graph: GraphRow, randomize_node_id=False):
    """Re-index the node ids from 0

    Args:
        graph ([type]): graph object
        randomize_node_id (bool): Defaults to False. If True,
            then apply randomized node ids

    Returns:
        [type]: re-indexed graph object
    """
    node_map = {}
    for e in graph.edges + graph.noise_edges:
        if e[0] not in node_map:
            node_map[e[0]] = len(node_map)
        if e[1] not in node_map:
            node_map[e[1]] = len(node_map)
    if randomize_node_id > 0:
        new_node_map = copy.deepcopy(node_map)
        node_keys = list(node_map.keys())
        rand_keys = random.sample(node_keys, len(node_keys))
        for ni, nk in enumerate(node_keys):
            new_node_map[rand_keys[ni]] = node_map[nk]
        node_map = copy.deepcopy(new_node_map)
    # reset
    new_graph_edges = []
    new_graph_noise_edges = []
    new_graph_resolution_path = []
    # replace
    for e in graph.edges:
        new_graph_edges.append((node_map[e[0]], node_map[e[1]], e[2]))
    for e in graph.noise_edges:
        new_graph_noise_edges.append((node_map[e[0]], node_map[e[1]], e[2]))
    for p in graph.resolution_path:
        new_graph_resolution_path.append(node_map[p])
    new_graph_query = [node_map[graph.query[0]], node_map[graph.query[1]]]
    new_graph = GraphRow(
        new_graph_edges,
        graph.source,
        graph.sink,
        graph.target,
        new_graph_query,
        graph.descriptor,
        graph.rules_used,
        graph.rules_used_pos,
        new_graph_resolution_path,
        new_graph_noise_edges,
    )
    return new_graph


def apply_noise_row(graph: GraphRow, args: DictConfig):
    """
    Apply noise on graphs based on the policy

    Available policies = "dangling" / "disconnected" / "supporting"

    - supporting: the source must have at least one outgoing node, and the sink must have one incoming node
     (although not guranteed that there exist a path in between)

    - disconnected: the source *should not* have any outgoing nodes and sink should not have any incoming nodes
    (guranteed disconnection)

    - dangling: any node in resolution path has an incoming or outgoing node

    """
    resolution_path = graph.resolution_path
    outgoing_edges = {}
    incoming_edges = {}
    any_path = False
    for node in resolution_path:
        outgoing_edges[node] = []
        for ne in graph.noise_edges:
            if ne[0] == node:
                outgoing_edges[node].append((ne[0], ne[1]))
                any_path = True
        incoming_edges[node] = []
        for ne in graph.noise_edges:
            if ne[1] == node:
                incoming_edges[node].append((ne[0], ne[1]))
                any_path = True
    noise_edges = graph.noise_edges
    # supporting
    if args.noise_policy == "supporting":
        if not (
            len(outgoing_edges[graph.source]) >= 1
            and len(incoming_edges[graph.sink]) >= 1
        ):
            raise AssertionError("supporting noise cannot be added to this graph!")
    # dangling
    elif args.noise_policy == "dangling":
        if any_path:
            # remove either all incoming or outgoing edges
            remove_choice = random.choice(["incoming", "outgoing"])
            if remove_choice == "incoming":
                remove_edges = outgoing_edges
            else:
                remove_edges = incoming_edges
            to_del = set(
                list([x for node in resolution_path for x in remove_edges[node]])
            )
            noise_edges = [
                edge for edge in graph.noise_edges if (edge[0], edge[1]) not in to_del
            ]
        else:
            raise AssertionError("dangling noise cannot be added to this graph!")
    # disconnected
    elif args.noise_policy == "disconnected":
        # remove all incoming and outgoing nodes from the resolution path
        to_del = set(
            list(
                [x for node in resolution_path for x in outgoing_edges[node]]
                + [x for node in resolution_path for x in incoming_edges[node]]
            )
        )
        noise_edges = [
            edge for edge in graph.noise_edges if (edge[0], edge[1]) not in to_del
        ]
    noise_graph = copy.deepcopy(graph)
    noise_graph.noise_edges = noise_edges
    return noise_graph


def auto_update_config(args: DictConfig) -> DictConfig:
    """
    Automatically update config based on user values
    """
    # Derive max_path_len from descriptor_lengths
    # max_path_len is always 1 more than the desired length of the puzzle
    max_path_len = (
        max(
            max([int(x) for x in args.train_descriptor_lengths.split(",")]),
            max([int(x) for x in args.val_descriptor_lengths.split(",")]),
            max([int(x) for x in args.test_descriptor_lengths.split(",")]),
        )
        + 1
    )
    args.max_path_len = max_path_len
    # Derive num_graphs. This controls the total number of graphs generated,
    # from which we will subsample
    max_num_graphs = max(
        [
            args.num_graphs,
            sum([args.num_train_graphs, args.num_valid_graphs, args.num_test_graphs]),
        ]
    )
    args.num_graphs = max_num_graphs
    return args


class GraphDataset:
    """
    Graph dataset object
    """

    def __init__(self, args: DictConfig, save_loc: Path) -> None:
        self.args = args
        self.rows: List[GraphRow] = []
        self.seen = {}
        self.ids = {}
        self.save_loc = save_loc

    def add_row(self, row: GraphRow):
        if self.args.re_index:
            row = re_index_nodes(row, self.args.randomize_node_id)
        row_str = row.get_str_without_noise()
        if not self.args.unique_graphs or row_str not in self.seen:
            self.rows.append(row)
            self.seen[row_str] = 1
            return True
        else:
            return False

    def set_split_ids(self, ids, split_type="train"):
        self.ids[split_type] = ids

    def __len__(self):
        return len(self.rows)

    def save(self):
        for split_type, split_ids in self.ids.items():
            graphs = []
            for split_id in split_ids:
                row = self.rows[split_id]
                graph = {
                    "edges": row.edges,
                    "source": row.source,
                    "sink": row.sink,
                    "query": row.query,
                    "target": row.target,
                    "descriptor": row.descriptor,
                    "rules_used": row.rules_used,
                    "rules_used_pos": row.rules_used_pos,
                    "resolution_path": row.resolution_path,
                    "noise_edges": row.noise_edges,
                }
                graphs.append(graph)
            save_to = self.save_loc / f"{split_type}.jsonl"
            print(f"Saving {len(graphs)} graphs in {save_to} ...")
            dump_jsonl(graphs, save_to)

    def apply_noise(self):
        noisy_rows = []
        issue_ct = 0
        for ri, row in enumerate(self.rows):
            try:
                row = apply_noise_row(row, self.args)
                row.edges = row.edges + row.noise_edges
                noisy_rows.append(row)
            except:
                issue_ct += 1
        if issue_ct > 0:
            issue_per = np.round(issue_ct / len(self.rows), 2) * 100
            print(
                f"Unable to apply noise policy '{self.args.noise_policy}' to {issue_per} % ({issue_ct}/{len(self.rows)}) rows."
            )
            print("Tip: Consider increasing the number of graphs to search!")
        self.rows = noisy_rows


def split_world(
    world: GraphDataset,
    test_size=0.2,
    keep_train_des_len=2,
    train_des_lens="",
    val_des_lens="",
    test_des_lens="",
):
    """Split the graphs

    Args:
        world ([type]): [description]
        test_size (float, optional): [description]. Defaults to 0.2.
        keep_train_des_len (int, optional): [description]. Defaults to 3. descriptors of length <=2 should be added to train set
        train_des_lens: comma separated descriptor lengths to include in train set
        val_des_lens: comma separated descriptor lengths to include in val set
        test_des_lens: comma separated descriptor lengths to include in test set

    Returns:
        [type]: [description]
    """
    graphs = world.rows
    des = list(Counter(graphs[i].descriptor for i in range(len(graphs))).keys())
    print(f"Generated {len(des)} unique descriptors")
    train_des = []
    res_des = []
    des_len_ct = {}
    for d in des:
        num_des = len(d.split(","))
        # always keep these descriptors in train distribution
        if num_des <= keep_train_des_len:
            train_des.append(d)
        else:
            res_des.append(d)
        if num_des not in des_len_ct:
            des_len_ct[num_des] = 0
        des_len_ct[num_des] += 1

    print(f"Descriptor distribution: {des_len_ct}")

    train_m_des, test_des = train_test_split(res_des, test_size=test_size)
    train_m_des, val_des = train_test_split(train_m_des, test_size=test_size)

    train_des = train_des + train_m_des
    val_des = set(val_des)
    test_des = set(test_des)

    # Filter
    train_des_lens = get_des_ids(train_des_lens)
    val_des_lens = get_des_ids(val_des_lens)
    test_des_lens = get_des_ids(test_des_lens)
    if len(train_des_lens) > 0:
        print("Filtering train descriptors ...")
        train_des = set(
            [des for des in train_des if len(des.split(",")) in train_des_lens]
        )
    if len(val_des_lens) > 0:
        print("Filtering val descriptors ...")
        val_des = set([des for des in val_des if len(des.split(",")) in val_des_lens])
    if len(test_des_lens) > 0:
        print("Filtering test descriptors ...")
        test_des = set(
            [des for des in test_des if len(des.split(",")) in test_des_lens]
        )

    def describe(des):
        return Counter([len(x.split(",")) for x in des])

    print(f"Train unique descriptors: Total: {len(train_des)}, {describe(train_des)}")
    print(f"Val unique descriptors: Total: {len(val_des)}, {describe(val_des)}")
    print(
        f"Test unique descriptors: Total: {len(test_des)}, Dist: {describe(test_des)}"
    )

    train_ids = []
    val_ids = []
    test_ids = []
    for i, rec in enumerate(graphs):
        if rec.descriptor in test_des:
            test_ids.append(i)
        elif rec.descriptor in val_des:
            val_ids.append(i)
        else:
            train_ids.append(i)
    return des, train_ids, val_ids, test_ids


@hydra.main(config_name="graph_config")
def main(args: DictConfig):
    set_seed(args.seed)
    args = auto_update_config(args)
    print(args)
    task = args.world_id
    ## Expects a dictionary of compositional rules, which contains rules of the form
    ## { (r_1, r_2): r_3 }, or (head -> body)
    if args.world_mode:
        save_loc = Path(args.save_loc) / f"{args.world_mode}/{args.world_prefix}_{task}"
    else:
        save_loc = Path(args.save_loc) / f"{args.world_prefix}_{task}"
    # create folder if not present
    save_loc.mkdir(exist_ok=True, parents=True)
    rules = verify_and_load_rule_store(
        Path(hydra.utils.get_original_cwd()) / f"{args.rule_store}_{task}.json"
    )
    print(f"Found {len(rules)} rules")
    graph_store = GraphDataset(args, save_loc)
    if args.unique_graphs:
        print(
            "Warning: unique constraints set, number of graphs found maybe lower than requested."
        )
    max_attempts = args.num_graphs * args.search_multiplier
    attempt = 0
    pb = tqdm(total=args.num_graphs)
    while len(graph_store) < args.num_graphs:
        graph_row = sample_graph(
            rules,
            max_path_len=args.max_path_len,
            num_steps=args.num_steps,
            expansion_prob=args.expansion_prob,
            num_completion_steps=args.num_completion_steps,
            debug=False,
        )
        attempt += 1
        if graph_store.add_row(graph_row):
            pb.update(1)
        if attempt > max_attempts:
            print(
                "Exhausted the number of attempts for graph search, consider lowering the total number of graphs requested."
            )
            break
    pb.close()
    # Apply noise
    # Till now we have kept a separate list of edges for noise which are not accessed
    # This is where we add them to our graphs depending on the noise addition policy
    graph_store.apply_noise()
    graph_store.save()

    # split train test
    rows_str = human_format(args.num_graphs)
    _, train_ids, val_ids, test_ids = split_world(
        graph_store,
        test_size=args.test_size,
        train_des_lens=args.train_descriptor_lengths,
        val_des_lens=args.val_descriptor_lengths,
        test_des_lens=args.test_descriptor_lengths,
    )
    print(
        f"Generated records: #Train: {len(train_ids)}, #Val: {len(val_ids)}, #Test: {len(test_ids)}"
    )
    print("Subsampling ...")
    # subsample if exact number required
    if len(train_ids) > args.num_train_graphs:
        train_ids = random.sample(train_ids, args.num_train_graphs)
    if len(val_ids) > args.num_valid_graphs:
        val_ids = random.sample(val_ids, args.num_valid_graphs)
    if len(test_ids) > args.num_test_graphs:
        test_ids = random.sample(test_ids, args.num_test_graphs)
    print(
        f"Storing records: #Train: {len(train_ids)}, #Val: {len(val_ids)}, #Test: {len(test_ids)}"
    )
    # Store splits
    json.dump(
        {"train": train_ids, "valid": val_ids, "test": test_ids},
        open(save_loc / f"splits_{rows_str}.json", "w"),
    )
    graph_store.set_split_ids(train_ids, split_type="train")
    graph_store.set_split_ids(val_ids, split_type="valid")
    graph_store.set_split_ids(test_ids, split_type="test")
    graph_store.save()
    # Save graphs
    # Sample world graph
    if args.world_graph.sample:
        world_edges = sample_world_graph(
            rule_world=rules,
            max_path_len=args.world_graph.max_path_len,
            num_sampled_paths=args.world_graph.num_sampled_paths,
            num_steps=args.world_graph.num_steps,
            expansion_prob=args.world_graph.expansion_prob,
            num_completion_steps=args.world_graph.num_completion_steps,
            debug=True,
        )
        world_graph = [{"edges": world_edges}]
        dump_jsonl(world_graph, save_loc / "meta_graph.jsonl")
    # Legacy - config.json
    legacy_rules = [
        {"body": body.split(","), "head": head, "p": 1.0}
        for body, head in rules.items()
    ]
    legacy_config = OmegaConf.to_container(args, resolve=True)
    legacy_config["rules"] = legacy_rules
    json.dump(legacy_config, open(save_loc / "config.json", "w"))


if __name__ == "__main__":
    main()
