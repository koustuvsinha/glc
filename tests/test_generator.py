## GraphLog v2.0 testing file
from glc import (
    apply_noise_row,
    get_incoming_outgoing_edges,
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
    graph = sample_graph(rule_world=rule_world, max_path_len=5, random_path_len=False)
    assert graph.source == graph.edges[0][0]
    assert graph.sink == graph.edges[-1][1]
    assert len(graph.rules_used) == len(graph.edges) - 1
    assert len(graph.rules_used) == len(graph.rules_used_pos)
    ## check target generated from first rule
    assert rule_world[graph.rules_used[0]] == graph.target
    ## Check total edges
    assert len(graph.edges) == 5
    assert type(graph.descriptor) == str
    assert len(graph.descriptor.split(",")) == len(graph.edges)
    assert len(graph.resolution_path) == len(graph.edges) + 1


def test_dangling_noise():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    graph = sample_graph(
        rule_world=rule_world,
        max_path_len=5,
        random_path_len=False,
        seed=10,
    )
    _, _, _ = get_incoming_outgoing_edges(graph)
    graph = apply_noise_row(graph, noise_policy="dangling", seed=10)
    (
        post_incoming_edges,
        post_outgoing_edges,
        _,
    ) = get_incoming_outgoing_edges(graph)
    num_outgoing_edges = sum([len(v) for k, v in post_outgoing_edges.items()])
    num_incoming_edges = sum([len(v) for k, v in post_incoming_edges.items()])
    assert num_outgoing_edges == 0 or num_incoming_edges == 0


def test_disconnected_noise():
    set_seed(42)
    rule_world = {"R_1,R_2": "R_3", "R_2,R_3": "R_1", "R_3,R_1": "R_2"}
    graph = sample_graph(
        rule_world=rule_world,
        max_path_len=5,
        random_path_len=False,
        seed=10,
    )
    _, _, _ = get_incoming_outgoing_edges(graph)
    graph = apply_noise_row(graph, noise_policy="disconnected", seed=10)
    (
        post_incoming_edges,
        post_outgoing_edges,
        _,
    ) = get_incoming_outgoing_edges(graph)
    num_outgoing_edges = sum([len(v) for k, v in post_outgoing_edges.items()])
    num_incoming_edges = sum([len(v) for k, v in post_incoming_edges.items()])
    assert num_outgoing_edges == 0 and num_incoming_edges == 0
