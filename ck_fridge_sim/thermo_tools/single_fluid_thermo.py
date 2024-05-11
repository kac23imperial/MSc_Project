import json
from abc import ABC
from typing import Optional

from CoolProp import CoolProp as CP

from ck_fridge_sim.thermo_tools.equations_of_state.equation_of_state_base import (
    EquationOfStateBase,
)
from ck_fridge_sim.thermo_tools.thermo_constants import ThermoConstants
from ck_fridge_sim.thermo_tools.unit_conversions import UnitConversions


class SingleFluidThermo(ABC):
    # FLUID should be overridden by child classes
    FLUID = None

    # EOS should be overridden by child classes
    EOS = EquationOfStateBase

    N_COMPONENTS = 1

    def __init__(
        self,
        pressure__pa=None,
        temp__c=None,
        quality=None,
        enthalpy__kj_kg=None,
        entropy__kj_kgk=None,
        density__kg_m3=None,
        fluid: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        temp__c : float
            Temperature in C
        quality : float
            Quality
        enthalpy__kj_kg : float
            Enthalpy in kJ/kg
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        density__kg_m3 : float
            Density in kg/m3
        fluid : str
            Name of the fluid if None, the default FLUID is used
        """
        self.FLUID = fluid if fluid else self.FLUID
        assert self.FLUID is not None, "FLUID or fluid must be specified"
        if self.FLUID not in CP.FluidsList():
            raise ValueError(f"Invalid FLUID: {self.FLUID}")

        assert issubclass(
            self.EOS, EquationOfStateBase
        ), "EOS must be overridden by child classes"
        self.eos = self.EOS()

        thermodynamic_state = self._validate_properties(
            pressure__pa,
            temp__c,
            quality,
            enthalpy__kj_kg,
            entropy__kj_kgk,
            density__kg_m3,
        )
        try:
            self._thermodynamic_state = self.eos.complete_thermodynamic_state(
                thermodynamic_state, self.FLUID
            )
        except Exception as ex:
            if verbose:
                print(ex)
                print(
                    f"Invalid thermodynamic state provided: {json.dumps(thermodynamic_state, indent=4)}"
                )

            raise ex

    @staticmethod
    def _validate_properties(
        pressure__pa, temp__c, quality, enthalpy__kj_kg, entropy__kj_kgk, density__kg_m3
    ):
        """
        This ensures that the number of degrees of freedom is consistent with the number of
        of phases that can exist for a given fluid.

        Parameters
        ----------
        pressure__pa : float
            Pressure in Pa
        temp__c : float
            Temperature in C
        quality : float
            Quality
        enthalpy__kj_kg : float
            Enthalpy in kJ/kg
        entropy__kj_kgk : float
            Entropy in kJ/kg-K
        density__kg_m3 : float
            Density in kg/m3

        Returns
        -------
        List(Tuple(str, float))
            List of tuples containing the specification and value
        """
        degrees_speficified = 0
        thermodynamic_state = {}
        for value, property_ in zip(
            (
                pressure__pa,
                temp__c,
                quality,
                enthalpy__kj_kg,
                entropy__kj_kgk,
                density__kg_m3,
            ),
            (ThermoConstants.PROPERTIES),
        ):
            if value is not None:
                degrees_speficified += 1
                thermodynamic_state[property_] = {
                    ThermoConstants.UNITS: ThermoConstants.PROP_UNIT_DICT[property_],
                    ThermoConstants.VALUE: value,
                }
                if property_ == ThermoConstants.QUALITY:
                    assert 0 <= value <= 1

        if degrees_speficified != 2:
            raise ValueError(
                f"Invalid number of degrees of freedom specified, got {degrees_speficified}, expected {2}"
            )

        return thermodynamic_state

    @property
    def thermodynamic_state(self):
        return self._thermodynamic_state

    @property
    def pressure__pa(self):
        return self.thermodynamic_state[ThermoConstants.PRESSURE][ThermoConstants.VALUE]

    @property
    def pressure__psig(self):
        return UnitConversions.pascal_to_psig(self.pressure__pa)

    @property
    def temp__c(self):
        return self.thermodynamic_state[ThermoConstants.TEMPERATURE][
            ThermoConstants.VALUE
        ]

    @property
    def temp__f(self):
        return UnitConversions.celsius_to_fahrenheit(self.temp__c)

    @property
    def quality(self):
        return self.thermodynamic_state[ThermoConstants.QUALITY][ThermoConstants.VALUE]

    @property
    def enthalpy__kj_kg(self):
        return self.thermodynamic_state[ThermoConstants.ENTHALPY][ThermoConstants.VALUE]

    @property
    def entropy__kj_kgk(self):
        return self.thermodynamic_state[ThermoConstants.ENTROPY][ThermoConstants.VALUE]

    @property
    def density__kg_m3(self):
        return self.thermodynamic_state[ThermoConstants.DENSITY][ThermoConstants.VALUE]

    @classmethod
    def molar_mass__kg_mol(cls):
        assert issubclass(
            cls.EOS, EquationOfStateBase
        ), "EOS must be overridden by child classes"
        cls.eos = cls.EOS()
        assert cls.FLUID is not None, "FLUID or fluid must be specified"
        return cls.eos.get_molar_mass(cls.FLUID)

    @property
    def molar_enthalpy__kj_mol(self):
        return self.enthalpy__kj_kg * self.molar_mass__kg_mol()

    @property
    def molar_entropy__kj_molk(self):
        return self.entropy__kj_kgk * self.molar_mass__kg_mol()

    @property
    def molar_density__mol_m3(self):
        return self.density__kg_m3 / self.molar_mass__kg_mol()
