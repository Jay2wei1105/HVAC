from engine.addons.registry import ADDON_REGISTRY


def test_registry_contains_second_domain_addon() -> None:
    assert "compressed_air" in ADDON_REGISTRY


def test_compressed_air_addon_instantiates() -> None:
    addon = ADDON_REGISTRY["compressed_air"]()
    assert addon.domain_id == "compressed_air"
    assert addon.prediction.get_prediction_targets() == ["total_power"]
