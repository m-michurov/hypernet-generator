from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TypeAlias, TypeVar, Iterable

import networkx as nx
from matplotlib import pyplot as plt

Vertex: TypeAlias = int
Edge: TypeAlias = frozenset[Vertex]
Path: TypeAlias = tuple[Vertex, ...]


@dataclass
class Graph:
    vertices: set[Vertex]
    edges: set[Edge]

    def copy(self) -> Graph:
        return Graph(vertices=self.vertices.copy(), edges=self.edges.copy())


@dataclass
class GraphEmbedding:
    primary: Graph
    secondary: Graph
    vertex_embedding: dict[Vertex, Vertex]
    edge_embedding: dict[Edge, Path]
    initial_edge_capacity: dict[Edge, int]
    remaining_edge_capacity: dict[Edge, int]


def path_to_edges(path: Path) -> list[Edge]:
    return [Edge((i, j)) for i, j in zip(path, path[1:])]


def edges_to_pairs(edges: Iterable[Edge]) -> list[tuple[Vertex, Vertex]]:
    result = []
    for edge in edges:
        assert len(edge) == 2
        i, j = sorted(edge)
        result.append((i, j))
    return result


T = TypeVar('T')


def choose(rng: random.Random, choices: set[T]) -> T:
    return rng.choice(list(choices))


def choose_and_remove(rng: random.Random, choices: set[T]) -> T:
    t = choose(rng, choices)
    choices.remove(t)
    return t


def choose_subset(rng: random.Random, choices: set[T], k: int) -> set[T]:
    return set(rng.sample(list(choices), k=k))


def get(s: set[T] | frozenset[T]) -> T:
    assert len(s) == 1
    return next(iter(s))


def generate_random_tree(rng: random.Random, n_vertices: int) -> Graph:
    assert n_vertices > 0

    vertices = set(range(n_vertices))
    root: Vertex = choose(rng, vertices)

    included_vertices: set[Vertex] = {root}
    available_vertices: set[Vertex] = vertices - included_vertices

    choices: set[Edge] = set(Edge((root, v)) for v in range(n_vertices) if v != root)

    edges: set[Edge] = set()

    while choices:
        edge = choose(rng, choices)
        edges |= {edge}

        j, = edge & available_vertices

        available_vertices -= {j}
        included_vertices |= {j}

        choices -= {Edge((i, j)) for i in included_vertices}
        choices |= {Edge((j, k)) for k in available_vertices}

    assert len(edges) == n_vertices - 1
    return Graph(vertices, edges)


def generate_random_connected_graph(rng: random.Random, vertices_count: int, edges_count: int) -> Graph:
    min_edges = vertices_count - 1
    max_edges = vertices_count * (vertices_count - 1) // 2
    assert min_edges <= edges_count <= max_edges, \
        f'A connected graph with {vertices_count} vertices can have between {min_edges} and {max_edges} edges, not {edges_count}.'

    spanning_tree = generate_random_tree(rng, vertices_count)

    choices: set[Edge] = {
        Edge((i, j))
        for i in spanning_tree.vertices for j in spanning_tree.vertices
        if i != j and Edge((i, j)) not in spanning_tree.edges
    }

    edges = spanning_tree.edges | choose_subset(rng, choices, k=edges_count - len(spanning_tree.edges))
    assert len(edges) == edges_count

    return Graph(spanning_tree.vertices, edges)


def is_path_allowed(path: Path, edge_capacity: dict[Edge, int]) -> bool:
    return all(edge_capacity[edge] > 0 for edge in path_to_edges(path))


def all_allowed_paths_between_vertices(graph: Graph, i: Vertex, j: Vertex, edge_capacity: dict[Edge, int]) -> set[Path]:
    graph_nx = nx.Graph()
    graph_nx.add_edges_from(edges_to_pairs(graph.edges))
    return {
        Path(path)
        for path in nx.all_simple_paths(graph_nx, i, j)
        if is_path_allowed(Path(path), edge_capacity)
    }


def all_allowed_paths(graph: Graph, edge_capacity: dict[Edge, int]) -> set[Path]:
    all_paths = set()
    for i in graph.vertices:
        for j in graph.vertices:
            if i == j:
                continue
            all_paths |= all_allowed_paths_between_vertices(graph, i, j, edge_capacity)

    return all_paths


def is_edge_embedding_allowed(edge: Edge, path: Path, vertex_embedding: dict[Vertex, Vertex]) -> bool:
    i, j = sorted(edge)

    if not (i not in vertex_embedding or vertex_embedding[i] == path[0]):
        return False

    if not (j not in vertex_embedding or vertex_embedding[j] == path[-1]):
        return False

    if not (path[0] not in vertex_embedding.values() or vertex_embedding.get(i) == path[0]):
        return False

    if not (path[-1] not in vertex_embedding.values() or vertex_embedding.get(j) == path[-1]):
        return False

    return True


def generate_random_embedding(
        rng: random.Random,
        primary: Graph,
        edge_capacity: dict[Edge, int],
        secondary: Graph
) -> tuple[GraphEmbedding | None, int, int]:
    assert len(primary.vertices) >= len(secondary.vertices)

    if not secondary.vertices:
        return None, 0, 0

    all_paths = all_allowed_paths(primary, edge_capacity)
    path_edges = {path: path_to_edges(path) for path in all_paths}

    @dataclass
    class Step:
        choices: set[tuple[Edge, Path]]
        edge_capacity: dict[Edge, int]
        vertex_embedding: dict[Vertex, Vertex]
        edge_embedding: dict[Edge, Path]

        def copy_next(self) -> Step:
            return Step(
                choices=self.choices.copy(),
                edge_capacity=self.edge_capacity.copy(),
                vertex_embedding=self.vertex_embedding.copy(),
                edge_embedding=self.edge_embedding.copy(),
            )

    steps: list[Step] = [Step(
        choices={(edge, path) for edge in secondary.edges for path in all_paths},
        edge_capacity=edge_capacity.copy(),
        vertex_embedding={},
        edge_embedding={},
    )]

    total_choices_explored = 0
    times_backtracked = 0

    while steps:
        while (step := steps[-1]).choices:
            total_choices_explored += 1

            edge, path = choose_and_remove(rng, step.choices)

            next_step = step.copy_next()

            next_step.edge_embedding[edge] = path

            i, j = sorted(edge)
            next_step.vertex_embedding[i] = path[0]
            next_step.vertex_embedding[j] = path[-1]

            for primary_edge in path_edges[path]:
                next_step.edge_capacity[primary_edge] -= 1

            exclude = {
                (other_edge, other_path)
                for (other_edge, other_path) in next_step.choices
                if other_edge in next_step.edge_embedding
                   or any(next_step.edge_capacity[path_edge] <= 0 for path_edge in path_edges[other_path])
                   or not is_edge_embedding_allowed(other_edge, other_path, next_step.vertex_embedding)
            }

            next_step.choices -= exclude

            steps.append(next_step)

        if len(steps[-1].edge_embedding.keys()) == len(secondary.edges):
            break

        steps.pop()
        times_backtracked += 1

    if not steps or len(steps[-1].edge_embedding.keys()) != len(secondary.edges):
        return None, total_choices_explored, times_backtracked

    final_step = steps[-1]

    assert set(final_step.edge_embedding.keys()) == secondary.edges
    assert set(final_step.vertex_embedding.keys()) == secondary.vertices
    assert len(final_step.vertex_embedding.values()) == len(secondary.vertices)

    remaining_capacity = edge_capacity.copy()
    for path in final_step.edge_embedding.values():
        for edge in path_edges[path]:
            remaining_capacity[edge] -= 1

    assert remaining_capacity == final_step.edge_capacity
    assert all(capacity >= 0 for capacity in remaining_capacity.values())

    embedding = GraphEmbedding(
        primary=primary,
        secondary=secondary,
        vertex_embedding=final_step.vertex_embedding,
        edge_embedding=final_step.edge_embedding,
        initial_edge_capacity=edge_capacity,
        remaining_edge_capacity=final_step.edge_capacity,
    )

    return embedding, total_choices_explored, times_backtracked


def draw_graph(
        graph: Graph,
        vertex_labels: dict[Vertex, str] | None = None,
) -> tuple[nx.Graph, dict]:
    g = nx.Graph()
    g.add_edges_from(edges_to_pairs(graph.edges))
    is_planar, _ = nx.check_planarity(g)
    if not is_planar:
        pos = nx.spring_layout(g, seed=0)
    else:
        pos = nx.planar_layout(g)

    nx.draw(g, pos=pos, node_size=600)
    nx.draw_networkx_labels(
        g, pos,
        font_size=10,
        font_color='white',
        labels=vertex_labels
    )

    return g, pos


def draw_path(g: nx.Graph, pos: dict, path: Path, edge_color: str, width: int) -> None:
    nx.draw_networkx_edges(
        g, pos,
        edgelist=edges_to_pairs(path_to_edges(path)),
        edge_color=edge_color,
        width=width
    )


def draw_edge_labels(g: nx.Graph, pos: dict, edge_labels: dict[tuple[Vertex, Vertex], str]) -> None:
    nx.draw_networkx_edge_labels(
        g, pos,
        font_size=10,
        edge_labels=edge_labels
    )


def make_embedding_labels(embedding: GraphEmbedding) -> tuple[dict[Vertex, str], dict[tuple[Vertex, Vertex], str]]:
    vertex_labels = \
        {
            primary_vertex: f'{primary_vertex}←∅'
            for primary_vertex in embedding.primary.vertices
        } | {
            primary_vertex: f'{primary_vertex}←{secondary_vertex}'
            for secondary_vertex, primary_vertex in embedding.vertex_embedding.items()
        }

    def used_capacity(edge_: Edge) -> int:
        return embedding.initial_edge_capacity[edge_] - embedding.remaining_edge_capacity[edge_]

    edge_labels = {
        edge: f'{used_capacity(Edge(edge))}/{embedding.initial_edge_capacity[Edge(edge)]}'
        for edge in edges_to_pairs(embedding.primary.edges)
    }
    assert len(edge_labels) == len(embedding.primary.edges)

    return vertex_labels, edge_labels


def visualize_embedding(embedding: GraphEmbedding) -> None:
    plt.title(f'Secondary')
    draw_graph(embedding.secondary)
    plt.show()

    vertex_labels, edge_labels = make_embedding_labels(embedding)

    plt.title(f'Primary')
    draw_graph(embedding.primary, vertex_labels=vertex_labels)
    plt.show()

    def random_color() -> str:
        return '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

    for i, edge in enumerate(embedding.secondary.edges):
        i, j = sorted(edge)
        plt.title(f'({i}, {j})→({', '.join(map(str, embedding.edge_embedding[edge]))})')
        g, pos = draw_graph(embedding.primary, vertex_labels=vertex_labels)
        draw_path(g, pos, embedding.edge_embedding[edge], edge_color=random_color(), width=3)
        draw_edge_labels(g, pos, edge_labels)
        plt.show()


def main() -> None:
    rng = random.Random(1)

    primary_vertices_count = 6
    primary_edges_count = 13
    primary = generate_random_connected_graph(
        rng,
        vertices_count=primary_vertices_count,
        edges_count=primary_edges_count
    )
    primary_edge_capacity = {edge: 1 for edge in primary.edges}

    secondary_vertices_count = 5
    secondary_edges_count = 10
    secondary = generate_random_connected_graph(
        rng,
        vertices_count=secondary_vertices_count,
        edges_count=secondary_edges_count
    )

    start = time.time()

    embedding, total_choices, times_backtracked = generate_random_embedding(
        rng, primary, primary_edge_capacity, secondary)

    end = time.time()
    elapsed = end - start
    choices_per_second = total_choices / elapsed

    if not embedding:
        print(f'Embedding does not exist '
              f'(explored {total_choices} choices in {elapsed:.2f}s, '
              f'backtracked {times_backtracked} times, '
              f'{choices_per_second:.2f} choices per second).')
        return

    print(f'Embedding found '
          f'(explored {total_choices} choices in {elapsed:.2f}s, '
          f'backtracked {times_backtracked} times, '
          f'{choices_per_second:.2f} choices per second).')
    visualize_embedding(embedding)


if __name__ == "__main__":
    main()
