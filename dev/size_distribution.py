"""
Size Distributions
==================

A Size distribution has mass but no assay.
Does the class manage this appropriately?
Can we create a partition by dividing two size distributions?

"""

import logging
import time
from datetime import timedelta
from pathlib import Path

# %%
# Initialise
# ----------
from elphick.mass_composition.utils.size_distribution import rosin_rammler, modified_rosin_rammler, gaudin_schuhmann, \
    lynch

logger = logging.getLogger(name=Path(__file__).name)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
_tic = time.time()

# create some sample psd data using the rossin-rammler function

passing = rosin_rammler()
print(passing)

passing = modified_rosin_rammler()
print(passing)

passing = gaudin_schuhmann()
print(passing)

passing = lynch()
print(passing)

logger.info(f'{Path(__file__).name} execution duration: {timedelta(seconds=time.time() - _tic)}')
