import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List

from elphick.mass_composition.utils.components import is_compositional


class VariableGroups(Enum):
    MASS = 'mass'
    MOISTURE = 'moisture'
    CHEMISTRY = 'chemistry'


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

    def col_to_var(self) -> Dict:
        return {v.column_name: v.name for v in self.variables}


@dataclass()
class Variable:
    name: str
    group: VariableGroups
    name_match: Optional[str] = None
    name_force: Optional[str] = None

    @property
    def column_name(self):
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
        self.config: Dict = config
        self.supplied: List[str] = supplied
        self.specified_map: Dict[str, str] = specified_map
        self.variables: List[Variable] = []

        self.variables.extend(self._detect_mass_vars())
        self.variables.append(self._detect_moisture_var())
        self.variables.extend(self._detect_chemistry_vars())

        self.vars_mass: VariableGroup = VariableGroup(name=VariableGroups.MASS.value,
                                                      variables=[v for v in self.variables if
                                                                 v.group == VariableGroups.MASS])
        self.var_moisture: VariableGroup = VariableGroup(name=VariableGroups.MASS.value,
                                                         variables=[v for v in self.variables if
                                                                    v.group == VariableGroups.MOISTURE])
        self.vars_chemistry: VariableGroup = VariableGroup(name=VariableGroups.MASS.value,
                                                           variables=[v for v in self.variables if
                                                                      v.group == VariableGroups.CHEMISTRY])
        self.vars: VariableGroup = VariableGroup(name='all_variables', variables=self.variables)

    def _detect_mass_vars(self) -> List[Variable]:
        res: List = []
        for v in ['mass_wet', 'mass_dry']:
            match = re.search(self.config[v]['search_regex'], '\n '.join(self.supplied),
                              flags=re.IGNORECASE | re.MULTILINE)
            variable: Variable = Variable(name=self.config[v]['standard_name'],
                                          name_match=str(match.group()) if match else None,
                                          name_force=self.specified_map[v] if v in self.specified_map.keys() else None,
                                          group=VariableGroups.MASS)
            res.append(variable)
        return res

    def _detect_moisture_var(self) -> Variable:
        v = 'moisture'
        match = re.search(self.config[v]['search_regex'], '\n'.join(self.supplied),
                          flags=re.IGNORECASE | re.MULTILINE)
        variable: Variable = Variable(name=self.config[v]['standard_name'],
                                      name_match=str(match.group()) if match else None,
                                      name_force=self.specified_map[v] if v in self.specified_map.keys() else None,
                                      group=VariableGroups.MOISTURE)
        return variable

    def _detect_chemistry_vars(self) -> List[Variable]:
        res: List = []
        chem_ignore: List[str] = ['H2O'] + self.config['chemistry']['ignore']
        chem_ignore = list(set(chem_ignore + [c.lower() for c in chem_ignore] + [c.upper() for c in chem_ignore]))
        chemistry_vars: Dict[str, str] = {k: v for k, v in
                                          is_compositional(self.supplied, strict=False).items() if
                                          v not in chem_ignore}
        for k, v in chemistry_vars.items():
            variable: Variable = Variable(name=v,
                                          name_match=k,
                                          group=VariableGroups.CHEMISTRY)
            res.append(variable)
        return res
