import pycolmap
print(pycolmap.__version__)
for f in (pycolmap.extract_features,pycolmap.match_exhaustive,pycolmap.incremental_mapping):
    print(f.__doc__)
print(pycolmap.IncrementalPipelineOptions())
