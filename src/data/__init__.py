"""Data layer: raw IO, point-in-time joins, accounting restatement, preprocessing.

Pipeline stage 1 (see notes/notes.txt). Responsibilities are split across modules
so that loading (IO + caching), joining (no-lookahead alignment), accounting
normalization, and regularization can evolve independently.

Typical flow::

    raw  = loaders.load_all(cfg)          # feather -> frames
    panel = joins.build_panel(raw, cfg)   # point-in-time joined panel (USD/local)
    panel = accounting.restate(panel, cfg)# synthetic IFRS-9/CECL/IFRS-17/LDTI
    panel = preprocess.regularize(panel, cfg)  # winsorize -> fill -> standardize
"""
