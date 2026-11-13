import numpy as np

from scripts.export_dashboard_data import scatter


def test_aggregated_scatter_preserves_class_counts_at_exact_coordinates():
    points = np.array([[0.5, -1.0], [0.5, -1.0], [0.5, -1.0], [2.0, 3.0]])
    categories = ["A", "A", "B", "A"]

    grouped = scatter(points, categories, aggregate=True)

    assert {(point["x"], point["y"], point["c"]): point["n"] for point in grouped} == {
        (0.5, -1.0, "A"): 2,
        (0.5, -1.0, "B"): 1,
        (2.0, 3.0, "A"): 1,
    }
