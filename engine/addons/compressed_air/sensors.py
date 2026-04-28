from engine.core.types import SensorDefinition, SensorRole, SensorType


COMPRESSED_AIR_SENSOR_SCHEMA: list[SensorDefinition] = [
    SensorDefinition(
        "header_pressure",
        SensorType.PRESSURE,
        SensorRole.OBSERVABLE,
        "bar",
        ["pressure", "header_pressure"],
        (0, 20),
        required=True,
    ),
    SensorDefinition(
        "compressor_freq",
        SensorType.FREQUENCY,
        SensorRole.CONTROLLABLE,
        "Hz",
        ["compressor_freq", "vfd_hz"],
        (20, 60),
        required=True,
    ),
    SensorDefinition(
        "total_power",
        SensorType.POWER,
        SensorRole.OUTPUT,
        "kW",
        ["total_power", "kw"],
        (0, 2000),
        required=True,
    ),
]
