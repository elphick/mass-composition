Mass_Composition 0.6.4 (2024-05-16)
===================================

Bugfix
------

- Added missing name attribute on the output edges. (#165)


Mass_Composition 0.6.3 (2024-05-16)
===================================

Bugfix
------

- Fixed DAG incorrect edge successors using name lookup from graph. (#163)


Mass_Composition 0.6.2 (2024-05-16)
===================================

Bugfix
------

- Second fix for DAG.run ValueError('generator already running') - managed result type. (160a)


Mass_Composition 0.6.1 (2024-05-16)
===================================

Feature
-------

- DAG progress bar now optional.

Bugfix
------

- Attempted fix for DAG.run ValueError('generator already running') (#160)


Mass_Composition 0.6.0 (2024-05-16)
===================================

Feature
-------

- Added support for splitting with function and sklearn estimator.
- BREAKING CHANGE: renamed apply_partition to split_by_partition for method name consistency. (#147)


Mass_Composition 0.5.2 (2024-05-16)
===================================

Bugfix
------

- Fixed error when setting n_jobs > 1 in DAG. (#151)


Feature
-------

- Added progressbar to  DAG. (#151)


Mass_Composition 0.5.1 (2024-05-15)
===================================

Bugfix
------

- DAG now seems robust, bug fixed. Streams used in Flowsheet. (#150)
- Fix for the badges on the Sphinx documentation.


Mass_Composition 0.5.0 (2024-05-15)
===================================

Feature
-------

- Renamed MCNetwork to Flowsheet for improved context. (#154)

Other Tasks
-----------

- Removed init files from examples and tests


Mass_Composition 0.4.12 (2024-05-15)
====================================

Feature
-------

- Added Stream object as a version of MassComposition with source_node and destination_node attributes. (#87)


Mass_Composition 0.4.11 (2024-05-14)
====================================

Other Tasks
-----------

- Removed default loggers. (#149)


Mass_Composition 0.4.10 (2024-05-13)
====================================

Feature
-------

- Added parallel support with progressbar to
  streams_from_dataframe and MCNetwork.from_dataframe. (#134)


Mass_Composition 0.4.9 (2024-05-12)
===================================

Other Tasks
-----------

- Added warning for multi-index >2 levels.  New test.
- format fix for example 501. (#107)


Mass_Composition 0.4.8 (2024-05-12)
===================================

Feature
-------

- Added TqdmParallel to utils.
- removed pyvista from dependencies. (#140)


Mass_Composition 0.4.7 (2024-05-12)
===================================

Other Tasks
-----------

- Updated github workflows (#139)


Mass_Composition 0.4.6 (2024-05-12)
===================================

Improved Documentation
----------------------

- Added change log using towncrier package (#141)

Other Tasks
-----------

- Renamed `test` directory to `tests` (#141)
