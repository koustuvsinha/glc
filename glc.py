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
    body_0_map: Dict[str, Tuple[int, int]],
    body_1_map: Dict[str, Tuple[int, int]],
    head_map: Dict[str, int],
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
    u, v, R = None, None, None
    for t in range(search_ct):
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
    edges: List[Tuple[int, int, str]], rules: Dict[Tuple[int, int], str]
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
            edge_set.add((x, v, R_j))

    if len(u_1) > 0:
        R_j = R
        x, _, R_i = random.choice(u_1)
        cand_rules = [rule for rule in rules if rule[0][1] == R_j and rule[0][0] == R_i]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_k = inject_rule[1]
            edge_set.add((x, v, R_k))

    if len(v_0) > 0:
        R_i = R
        _, x, R_j = random.choice(v_0)
        cand_rules = [rule for rule in rules if rule[0][0] == R_i and rule[0][1] == R_j]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_k = inject_rule[1]
            edge_set.add((u, x, R_k))

    if len(v_1) > 0:
        R_k = R
        x, _, R_j = random.choice(v_1)
        cand_rules = [rule for rule in rules if rule[1] == R_k and rule[0][1] == R_j]
        if len(cand_rules) > 0:
            inject_rule = random.choice(cand_rules)
            R_i = inject_rule[0][0]
            edge_set.add((u, x, R_i))

    edges = list(edge_set)

    return edges


def sample_graph(
    rule_world: Dict[str, str],
    max_path_len: int = 5,
    add_noise: bool = True,
    num_steps: int = 50,
    expansion_prob: float = 0.5,
    num_completion_steps: int = 10,
    debug: bool = False,
    random_path_len: bool = True,
) -> Tuple[
    List[Tuple[int, int, str]], int, int, str, str, List[Any], List[int], List[int]
]:
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

    Returns:
        edges: list of edges of the graph (u,v,r)
        source: source node id
        sink: sink node id
        target: target relation to predict
        descriptor: descriptor of the graph, which is the concatenation of the relations in the shortest path from source to sink.
        rules_used: rules used to derive to the shortest path
        path: shortest path from the source to sink
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
    if add_noise:
        if debug:
            print("Sampling neighbors ...")
        new_node = sink + 1
        for _ in range(num_steps):
            if random.uniform(0, 1) > expansion_prob:
                edges = expansion_step(
                    edges, body_0_map, body_1_map, head_map, new_node
                )
            new_node += 1
            # run completion step for n numbers
            for _ in range(random.choice(range(1, num_completion_steps))):
                edges = completion_step(edges, rules)
    if debug:
        print("Saving edge to rel map ...")
    edge2rel = {(e[0], e[1]): e[2] for e in edges}
    g = nx.DiGraph()
    g.add_edges_from([(e[0], e[1]) for e in edges])
    if debug:
        print(nx.info(g))

    if debug:
        print("Removing shortest paths ...")
    ## If noise has been added, Compute and remove shortest paths
    if add_noise:
        sub_e = list(g.edges)
        aps = list(nx.all_simple_paths(g, source, sink, cutoff=path_length))
        # print(len(aps))
        sub_e = del_simple_paths(sub_e, aps, path)
        g = nx.from_edgelist(sub_e, create_using=nx.DiGraph)
        # print('Computing shortest paths from sink to source')
        aps = list(nx.all_simple_paths(g, path[-1], path[0], cutoff=path_length))
        # print(len(aps))
        sub_e = del_simple_paths(sub_e, aps, path)
        sub_e.extend(list(get_edges(path)))
    else:
        sub_e = list(g.edges)

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
    descriptor = ",".join(sampled_rule)
    return edges, source, sink, target, descriptor, rules_used, rules_used_pos, path


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


def split_world(
    world,
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
    des = list(Counter(world[i]["descriptor"] for i in range(len(world))).keys())
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
    for i, rec in enumerate(world):
        if rec["descriptor"] in test_des:
            test_ids.append(i)
        elif rec["descriptor"] in val_des:
            val_ids.append(i)
        else:
            train_ids.append(i)
    return des, train_ids, val_ids, test_ids


def re_index_nodes(graph, randomize_node_id=False):
    """Re-index the node ids from 0

    Args:
        graph ([type]): graph object
        randomize_node_id (bool): Defaults to False. If True,
            then apply randomized node ids

    Returns:
        [type]: re-indexed graph object
    """
    node_map = {}
    for e in graph["edges"]:
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
    new_graph = copy.deepcopy(graph)
    # reset
    new_graph["edges"] = []
    new_graph["resolution_path"] = []
    # replace
    for e in graph["edges"]:
        new_graph["edges"].append([node_map[e[0]], node_map[e[1]], e[2]])
    for p in graph["resolution_path"]:
        new_graph["resolution_path"].append(node_map[p])
    new_graph["query"][0] = node_map[graph["query"][0]]
    new_graph["query"][1] = node_map[graph["query"][1]]
    return new_graph


@hydra.main(config_name="graph_config")
def main(args: DictConfig):
    set_seed(args.seed)
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
    graph_store = []
    for _ in tqdm(range(args.num_graphs)):
        (
            edges,
            source,
            sink,
            target,
            descriptor,
            rules_used,
            rules_used_pos,
            path,
        ) = sample_graph(
            rules,
            max_path_len=args.max_path_len,
            add_noise=args.add_noise,
            num_steps=args.num_steps,
            expansion_prob=args.expansion_prob,
            num_completion_steps=args.num_completion_steps,
            debug=False,
        )
        graph = {
            "edges": edges,
            "query": [source, sink, target],
            "target": target,
            "descriptor": descriptor,
            "rules_used": rules_used,
            "rules_used_pos": rules_used_pos,
            "resolution_path": path,
        }
        if args.re_index:
            graph = re_index_nodes(graph, args.randomize_node_id)
        graph_store.append(graph)
    rows_str = human_format(args.num_graphs)
    dump_jsonl(graph_store, save_loc / f"graphs_{rows_str}_{task}.jsonl")
    # split train test
    _, train_ids, val_ids, test_ids = split_world(
        graph_store,
        test_size=args.test_size,
        train_des_lens=args.train_descriptor_lengths,
        val_des_lens=args.val_descriptor_lengths,
        test_des_lens=args.test_descriptor_lengths,
    )
    # subsample if exact number required
    if len(train_ids) > args.num_train_graphs:
        train_ids = random.sample(train_ids, args.num_train_graphs)
    if len(val_ids) > args.num_valid_graphs:
        val_ids = random.sample(val_ids, args.num_valid_graphs)
    if len(test_ids) > args.num_test_graphs:
        test_ids = random.sample(test_ids, args.num_test_graphs)
    # Store splits
    json.dump(
        {"train": train_ids, "valid": val_ids, "test": test_ids},
        open(save_loc / f"splits_{rows_str}.json", "w"),
    )
    # Store train/valid/test in separate files
    train_graphs = [graph_store[i] for i in train_ids]
    valid_graphs = [graph_store[i] for i in val_ids]
    test_graphs = [graph_store[i] for i in test_ids]
    dump_jsonl(train_graphs, save_loc / "train.jsonl")
    dump_jsonl(valid_graphs, save_loc / "valid.jsonl")
    dump_jsonl(test_graphs, save_loc / "test.jsonl")
    # Sample world graph
    if args.world_graph.sample:
        world_edges = sample_world_graph(
            rule_world=rules,
            max_path_len=args.world_graph.max_path_len,
            add_noise=args.add_noise,
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
