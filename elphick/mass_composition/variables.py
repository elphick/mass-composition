import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List

from elphick.mass_composition.utils.components import is_compositional


class VariableGroups(Enum):
    MASS = 'mass'
    MOISTURE = 'moisture'
    CHEMISTRY = 'chemistry'
    SUPPLEMENTARY = 'supplementary'


@dataclass()
class VariableGroup:
    name: str
    variables: List['Variable'] = field(default_factory=list)

    def get_var_names(self):
        return [v.name for v in self.variables]

    def get_col_names(self):
        return [v.column_name for v in self.variables]

    def var_to_col(self) -> Dict:
        return {v.name: v.column_name for v in self.variables}

    def var_to_format(self) -> Dict:
        return {v.name: v.format for v in self.variables}

    def col_to_var(self) -> Dict:
        return {v.column_name: v.name for v in self.variables}

    def col_to_format(self) -> Dict:
        return {v.column_name: v.format for v in self.variables}

    def property_to_var(self) -> Dict:
        return {v.name_property: v.name for v in self.variables}

    @classmethod
    def from_variable_groups(cls, name: str, variable_groups: List['VariableGroup']):
        variables: List = []
        for vg in variable_groups:
            variables.extend(vg.variables)
        return cls(name=name, variables=variables)


@dataclass()
class Variable:
    name: str  # the variable name
    name_property: str  # mass_wet, mass_dry, moisture, ...
    group: VariableGroups  # group that the variable belongs to
    format: Optional[str] = None  # display format string
    name_match: Optional[str] = None  # name found by the search
    name_force: Optional[str] = None  # name forced by user specification

    @property
    def column_name(self):  # the "go-to" name
        return self.name_force if self.name_force else self.name_match


class Variables:
    def __init__(self, config: Dict, supplied: List[str],
                 specified_map: Optional[Dict[str, str]] = None):
        """Initialise the Variables

        Args:
            config: The config defining the standard names and the search regex
            supplied: The supplied variables - the columns in the DataFrame
            specified_map: The variables specified by the user (will take precedence over any detected variables)
        """
        self._config: Dict = config
        self._supplied: List[str] = supplied
        self._specified_map: Dict[str, str] = specified_map
        self._variables: List[Variable] = []

        self._variables.extend(self._detect_mass_vars())
        self._variables.append(self._detect_moisture_var())
        self._variables.extend(self._detect_chemistry_vars())

        self.mass: VariableGroup = VariableGroup(name=VariableGroups.MASS.value,
                                                 variables=[v for v in self._variables if
                                                            v.group == VariableGroups.MASS])
        self.moisture: VariableGroup = VariableGroup(name=VariableGroups.MOISTURE.value,
                                                     variables=[v for v in self._variables if
                                                                v.group == VariableGroups.MOISTURE])
        self.mass_moisture: VariableGroup = VariableGroup.from_variable_groups(name='mass_moisture',
                                                                               variable_groups=[self.mass,
                                                                                                self.moisture])
        self.chemistry: VariableGroup = VariableGroup(name=VariableGroups.CHEMISTRY.value,
                                                      variables=[v for v in self._variables if
                                                                 v.group == VariableGroups.CHEMISTRY])
        self.core: VariableGroup = VariableGroup.from_variable_groups(name='core_variables',
                                                                      variable_groups=[self.mass,
                                                                                       self.chemistry])
        self.core_plus_moisture: VariableGroup = VariableGroup.from_variable_groups(name='core_variables',
                                                                                    variable_groups=[self.mass,
                                                                                                     self.moisture,
                                                                                                     self.chemistry])

        self._variables.extend(self._detect_supplementary_vars())

        self.supplementary: VariableGroup = VariableGroup(name=VariableGroups.SUPPLEMENTARY.value,
                                                          variables=[v for v in self._variables if
                                                                     v.group == VariableGroups.SUPPLEMENTARY])

        self.xr: VariableGroup = VariableGroup.from_variable_groups(name='xr_variables',
                                                                    variable_groups=[self.core,
                                                                                     self.supplementary])

        self.vars: VariableGroup = VariableGroup(name='all_variables', variables=self._variables)

    def _detect_mass_vars(self) -> List[Variable]:
        res: List = []
        for v in ['mass_wet', 'mass_dry']:
            match = re.search(self._config[v]['search_regex'], '\n '.join(self._supplied),
                              flags=re.IGNORECASE | re.MULTILINE)
            variable: Variable = Variable(name=self._config[v]['standard_name'],
                                          name_property=v,
                                          format=self._config[v]['format'],
                                          name_match=str(match.group()) if match else None,
                                          name_force=self._specified_map[
                                              f"{v}_var"] if f"{v}_var" in self._specified_map.keys() else None,
                                          group=VariableGroups.MASS)
            res.append(variable)
        return res

    def __eq__(self, other) -> bool:
        if isinstance(other, Variables):
            # Compare the instances based on their attributes
            return (self._variables == other._variables and self._config == other._config and
                    self._supplied == other._supplied and self._specified_map == other._specified_map)
        return False

    def _detect_moisture_var(self) -> Variable:
        v = 'moisture'
        match = re.search(self._config[v]['search_regex'], '\n'.join(self._supplied),
                          flags=re.IGNORECASE | re.MULTILINE)
        variable: Variable = Variable(name=self._config[v]['standard_name'],
                                      name_property=v,
                                      format=self._config[v]['format'],
                                      name_match=str(match.group()) if match else None,
                                      name_force=self._specified_map[
                                          f"{v}_var"] if f"{v}_var" in self._specified_map.keys() else None,
                                      group=VariableGroups.MOISTURE)
        return variable

    def _detect_chemistry_vars(self) -> List[Variable]:
        res: List = []
        chem_ignore: List[str] = ['H2O'] + self._config['chemistry']['ignore']
        chem_ignore = list(set(chem_ignore + [c.lower() for c in chem_ignore] + [c.upper() for c in chem_ignore]))
        chemistry_vars: Dict[str, str] = {k: v for k, v in
                                          is_compositional(self._supplied, strict=False).items() if
                                          v not in chem_ignore}
        for k, v in chemistry_vars.items():
            variable: Variable = Variable(name=v,
                                          name_property=v,
                                          format=self._config['chemistry']['format'],
                                          name_match=k,
                                          group=VariableGroups.CHEMISTRY)
            res.append(variable)
        return res

    def _detect_supplementary_vars(self) -> List[Variable]:
        res: List = []
        sup_vars: List = [col for col in self._supplied if col not in self.core_plus_moisture.get_col_names()]
        for v in sup_vars:
            variable: Variable = Variable(name=v,
                                          name_property=v,
                                          name_match=v,
                                          group=VariableGroups.SUPPLEMENTARY)
            res.append(variable)
        return res
