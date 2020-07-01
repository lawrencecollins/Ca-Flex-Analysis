# Plate Map #

Plate map uploading, processing & visualisaion.

empty_map() generates an empty well plate map according to a defined size:
6:(2, 3), 12:(3, 4), 24:(4, 6), 48:(6, 8), 96:(8, 12), 384:(16, 24)

plate_map() generates a full plate map from a 'long hand' table.

short_map() generates a full plate map from a 'short hand' table.

visualise() generates an image of the plate map described from the above functions.

readandmap() generates dataframes of the assay data associated with a plate map. 