from .hvac.addon import HVACAddon
from .compressed_air.addon import CompressedAirAddon

ADDON_REGISTRY: dict[str, type] = {
    "hvac": HVACAddon,
    "compressed_air": CompressedAirAddon,
}
