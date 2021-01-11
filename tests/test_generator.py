## GraphLog v2.0 testing file
from glc import (
    set_seed,
    create_maps,
    sample_path_and_target,
    expansion_step,
    completion_step,
    sample_graph,
)


def test_create_maps():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    rules = [(body.split(","), head) for body, head in rule_world.items()]
    body_0_map, body_1_map, head_map = create_maps(rules)
    assert len(body_0_map["R_1"]) == 1
    assert body_0_map["R_1"][0][0][0] == "R_1"
    assert len(body_1_map["R_1"]) == 1
    assert body_1_map["R_1"][0][0][1] == "R_1"
    assert len(head_map["R_1"]) == 1
    assert head_map["R_1"][0][1] == "R_1"


def test_sample_path_and_target():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    # Fix randomization of graph length using random_path_len = False
    (
        edges,
        rules_used,
        rules_used_pos,
        source,
        sink,
        target,
        sampled_rule,
    ) = sample_path_and_target(
        rule_world=rule_world, max_path_len=5, random_path_len=False
    )
    assert len(sampled_rule) == len(edges)
    assert source == edges[0][0]
    assert sink == edges[-1][1]
    assert len(rules_used) == len(edges) - 1
    assert len(rules_used) == len(rules_used_pos)
    ## check target generated from first rule
    assert rule_world[rules_used[0]] == target
    ## Check total edges
    assert len(edges) == 5
    ## Check logical validity of target
    for pos in rules_used_pos[::-1]:
        body = ",".join(sampled_rule[pos : pos + 2])
        t = rule_world[body]
        del sampled_rule[pos + 1]
        sampled_rule[pos] = t
    assert len(sampled_rule) == 1
    assert sampled_rule[0] == target


def test_expansion_step():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    rules = [(body.split(","), head) for body, head in rule_world.items()]
    body_0_map, body_1_map, head_map = create_maps(rules)
    (
        edges,
        rules_used,
        rules_used_pos,
        source,
        sink,
        target,
        sampled_rule,
    ) = sample_path_and_target(
        rule_world=rule_world, max_path_len=2, random_path_len=False
    )
    new_node = sink + 1
    num_edges = len(edges)
    edges = expansion_step(edges, body_0_map, body_1_map, head_map, new_node)
    num_ex_edges = len(edges)
    assert num_ex_edges == num_edges + 2


def test_completion_step():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    rules = [(body.split(","), head) for body, head in rule_world.items()]
    body_0_map, body_1_map, head_map = create_maps(rules)
    (
        edges,
        rules_used,
        rules_used_pos,
        source,
        sink,
        target,
        sampled_rule,
    ) = sample_path_and_target(
        rule_world=rule_world, max_path_len=6, random_path_len=False
    )
    new_node = sink + 1
    num_edges = len(edges)
    assert num_edges == 6
    edges = completion_step(edges, rules)
    num_cp_edges = len(edges)
    assert num_cp_edges == num_edges + 1


def test_sample_graph_no_noise():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    # Fix randomization of graph length using random_path_len = False
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
        rule_world=rule_world, max_path_len=5, add_noise=False, random_path_len=False
    )
    assert source == edges[0][0]
    assert sink == edges[-1][1]
    assert len(rules_used) == len(edges) - 1
    assert len(rules_used) == len(rules_used_pos)
    ## check target generated from first rule
    assert rule_world[rules_used[0]] == target
    ## Check total edges
    assert len(edges) == 5
    assert type(descriptor) == str
    assert len(descriptor.split(",")) == len(edges)
    assert len(path) == len(edges) + 1
