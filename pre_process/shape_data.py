from shapely import geometry
import shapefile
import pickle
import pandas as pd

# Read the shapefile of boroughs and create neighborhood data
sf = shapefile.Reader("data\London-wards-2014\London-wards-2014_ESRI\London_Ward_CityMerged.shp")

shapes = sf.shapes()
brgs_shp_raw = pd.DataFrame(sf.records())

# brgs_shp = extract_neigs(shapes, brgs_shp_raw)

brgs = brgs_shp_raw[5].unique()
brgs_shp = {}
err = []
for brg in brgs:
    idx = brgs_shp_raw[brgs_shp_raw[5] == brg].index.tolist()
    to_join = [geometry.Polygon(shapes[i].points) for i in idx]
    acc = to_join[0]
    if len(to_join) > 1:
        for i in to_join[1:]:
            if i.is_valid:
                acc = acc.union(i)
            else:
                err.append(brg)
    brgs_shp[brg] = acc

# Dict where the keys are the boroughs and each of them has values of the
# neighboring borough names
br_neigs = {brg: [i for i in brgs if (i != brg) & (brgs_shp[brg].distance(brgs_shp[i]) < 200)] for brg in brgs}

pickle.dump(br_neigs, open('data/br_neigs.pickle', 'wb'))
