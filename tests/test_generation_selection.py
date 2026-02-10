from layer_toolkit.generation.layers import (
    _build_summary_search_kwargs,
    _select_preferred_doc,
)


def test_build_summary_search_kwargs_with_material_id() -> None:
    kwargs = _build_summary_search_kwargs(
        element="Fe",
        spacegroup_number=229,
        material_id="mp-13",
        require_stable=True,
        max_energy_above_hull=0.02,
    )

    assert kwargs["material_ids"] == ["mp-13"]
    assert "elements" not in kwargs
    assert "spacegroup_number" not in kwargs
    assert kwargs["is_stable"] is True
    assert kwargs["energy_above_hull"] == (0.0, 0.02)


def test_build_summary_search_kwargs_with_filters_only() -> None:
    kwargs = _build_summary_search_kwargs(
        element="Fe",
        spacegroup_number=229,
        material_id=None,
        require_stable=False,
        max_energy_above_hull=None,
    )

    assert kwargs["elements"] == ["Fe"]
    assert kwargs["spacegroup_number"] == 229
    assert "material_ids" not in kwargs
    assert "is_stable" not in kwargs
    assert "energy_above_hull" not in kwargs


def test_select_preferred_doc_prefers_lowest_hull_then_material_id() -> None:
    docs = [
        {"material_id": "mp-200", "energy_above_hull": 0.08},
        {"material_id": "mp-100", "energy_above_hull": 0.08},
        {"material_id": "mp-050", "energy_above_hull": 0.01},
        {"material_id": "mp-999", "energy_above_hull": None},
    ]

    selected = _select_preferred_doc(docs)
    assert selected["material_id"] == "mp-050"
