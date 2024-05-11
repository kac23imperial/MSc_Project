class ThermoConstants:
    PRESSURE = "pressure"
    PRESSURE_UNITS = "pascals"
    TEMPERATURE = "temperature"
    TEMPERATURE_UNITS = "celsius"
    QUALITY = "quality"
    QUALITY_UNITS = "fraction"
    ENTHALPY = "enthalpy"
    ENTHALPY_UNITS = "kJ_kg"
    ENTROPY = "entropy"
    ENTROPY_UNITS = "kJ_kg-K"
    DENSITY = "density"
    DENSITY_UNITS = "kg_m3"
    PROP_UNIT_DICT = {
        PRESSURE: PRESSURE_UNITS,
        TEMPERATURE: TEMPERATURE_UNITS,
        QUALITY: QUALITY_UNITS,
        ENTHALPY: ENTHALPY_UNITS,
        ENTROPY: ENTROPY_UNITS,
        DENSITY: DENSITY_UNITS,
    }
    PROPERTIES = [PRESSURE, TEMPERATURE, QUALITY, ENTHALPY, ENTROPY, DENSITY]

    UNITS = "units"
    VALUE = "value"
