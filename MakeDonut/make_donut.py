#
# 
# MakeDonut: Polygon Splitting for EUDR Compliance
#
Description = '''
The MakeDonut function is designed to process polygons that contain interior holes, in accordance with the EUDR GeoJSON File Description, Version 1.5 (dated 5 May 2025, EUDR-API for EO Specification).

When a polygon with a hole is detected:

1. The algorithm computes its Minimum Rotated Rectangle (MRR) to determine the longitudinal axis and the perpendicular axis.

2. Using the perpendicular axis, the polygon is split into two simple polygons (without interior holes).

3. 3. 3. This procedure ensures proper handling of “donut” geometries for downstream compliance and geolocation verification workflows.

Purpose
This method supports EUDR requirements by standardizing polygon representations, enabling consistent geospatial analysis and ensuring that complex geometries are simplified into forms suitable for compliance assessment.
'''
# Author. Dr. Phisan Santitamnont , phisan.s@cdg.co.th , phisan.chula@gmail.com
# History.  version 0.1 (21 Aug 2025 )
# 
#
#
#
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import transform, split
from shapely.geometry.base import BaseGeometry
from shapely.affinity import translate
from shapely import wkt

import matplotlib.pyplot as plt

def drop_z(geom):
    if geom.is_empty:
        return geom
    # handle polygons
    if geom.geom_type == "Polygon":
        exterior = [(x, y) for x, y, *_ in geom.exterior.coords]
        interiors = [[(x, y) for x, y, *_ in ring.coords] for ring in geom.interiors]
        return Polygon(exterior, interiors)
    # handle multipolygons
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([drop_z(p) for p in geom.geoms])
    return geom  # fallback

def InsertPoly_with_Hole( gdf, idx_main, idx_hole ):
    poly_main = gdf.loc[idx_main, "geometry"]
    poly_hole = gdf.loc[idx_hole, "geometry"]
    # construct new polygon with hole
    poly_with_hole = Polygon(
        poly_main.exterior.coords,          # keep outer boundary
        [poly_hole.exterior.coords]         # add hole(s) as list
    )
     
    new_row = gdf.iloc[[0]].copy()
    new_row.loc[:,['part', "geometry"]] = [['outer_with_hole', poly_with_hole]]
    gdf = pd.concat([gdf, new_row], ignore_index=True)
    return gdf

def mrr_axes(poly: BaseGeometry, perp_len: float = 1.0, use_rect_centroid: bool = True):
    """
    Returns:
      mrr:        Polygon (minimum rotated rectangle)
      long_axis:  LineString through centroid, oriented along the MRR long side (full long length)
      perp_axis:  LineString through centroid, perpendicular to long_axis (total length = perp_len)
    """
    if poly.is_empty:
        raise ValueError("Empty geometry")

    # If MultiPolygon -> take largest part
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    elif poly.geom_type != "Polygon":
        raise TypeError(f"Expected Polygon/MultiPolygon, got {poly.geom_type}")

    # 1) Minimum rotated rectangle
    mrr = poly.minimum_rotated_rectangle

    # 2) Find long side direction
    corners = np.array(mrr.exterior.coords)[:4]                 # 4 unique corners
    edges = np.roll(corners, -1, axis=0) - corners              # vectors for the 4 edges
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    i_long = int(np.argmax(lengths))
    long_vec = edges[i_long]
    long_len = float(lengths[i_long])

    # Centroid (of rectangle by default)
    c = (mrr.centroid if use_rect_centroid else poly.centroid)
    cx, cy = float(c.x), float(c.y)

    # Unit vectors: long + perpendicular
    u_long = long_vec / long_len
    u_perp = np.array([-u_long[1], u_long[0]])

    # 3) Build axes lines through centroid
    # Longitudinal axis spans the full long side length
    p1_long = (cx - u_long[0]*long_len/2, cy - u_long[1]*long_len/2)
    p2_long = (cx + u_long[0]*long_len/2, cy + u_long[1]*long_len/2)
    long_axis = LineString([p1_long, p2_long])

    # Perpendicular axis has *total* length = perp_len
    p1_perp = (cx - u_perp[0]*perp_len/2, cy - u_perp[1]*perp_len/2)
    p2_perp = (cx + u_perp[0]*perp_len/2, cy + u_perp[1]*perp_len/2)
    perp_axis = LineString([p1_perp, p2_perp])

    return mrr, long_axis, perp_axis

def DoPlot(gdf):
    gdf = gdf.copy()
    #import pdb ; pdb.set_trace()
    g_outer = gdf.loc[gdf["part"].eq("outer")]
    g_hole  = gdf.loc[gdf["part"].eq("hole")]
    g_owh   = gdf.loc[gdf["part"].eq("outer_with_hole")]
    g_mrr   = gdf.loc[gdf["part"].eq("MRR")]
    g_d1    = gdf.loc[gdf["part"].eq("donut-1")]
    g_d2    = gdf.loc[gdf["part"].eq("donut-2")]
    GAP=2/111_000 
    g_d1.loc[:, "geometry"] = g_d1["geometry"].apply(
                    lambda g: translate(g, xoff=-GAP, yoff=0)).copy()
    g_d2.loc[:, "geometry"] = g_d2["geometry"].apply(
                    lambda g: translate(g, xoff=+GAP, yoff=0)).copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.ravel()

    def PolyLabel(gdf, ax, color):
        for idx, row in gdf.iterrows():
            x, y = row.geometry.centroid.coords[0]
            ax.text(x, y, str(idx+1), ha="center", va="center", 
                    fontsize=30, color=color)
    # 1st view: outer (red), hole (blue)
    if not g_outer.empty: g_outer.plot(ax=ax[0], color="none", ec="red",  lw=1.5)
    if not g_hole.empty:  g_hole.plot( ax=ax[0], color="none", ec="green", lw=1.5)
    PolyLabel(g_outer,ax[0],'red'); PolyLabel(g_hole,ax[0],'green' ) 
    ax[0].set_title("2 x Polygon (outer+hole)")
    # 2nd view: outer_with_hole (pink)
    if not g_owh.empty: g_owh.plot(ax=ax[1], color="none", ec="pink", lw=1.5)
    PolyLabel(g_owh,ax[1],'pink') 
    ax[1].set_title("1 x MultiPolygon")
    # 3rd view: outer_with_hole (pink) + MRR (black)
    if not g_owh.empty: g_owh.plot(ax=ax[2], color="none", ec="pink", lw=1.5)
    if not g_mrr.empty: g_mrr.plot(ax=ax[2], color="none", ec="black", lw=1.5)
    PolyLabel(g_owh,ax[2],'pink'); PolyLabel(g_mrr,ax[2],'black' ) 
    ax[2].set_title("MultiPolygon (pink) + MRR (red)")
    # 4th view: donut-1 (red), donut-2 (green)
    if not g_d1.empty: g_d1.plot(ax=ax[3], color="none", ec="red",   lw=1.5)
    if not g_d2.empty: g_d2.plot(ax=ax[3], color="none", ec="green", lw=1.5)
    PolyLabel(g_d1,ax[3],'red'); PolyLabel(g_d2,ax[3],'green' ) 
    ax[3].set_title("donut-1 (red) + donut-2 (green)")
    #########################################################################
    # common formatting
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("Longitude (°)")
        a.set_ylabel("Latitude (°)")
        a.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    #plt.show()
    PLT = 'MakeDonut.png'
    print(f'Ploting {PLT}...')
    plt.savefig( PLT )

######################################################################
######################################################################
######################################################################
GDB = 'ALRO_68_ST.gdb',  "wgs_eudrdb_eudrusr_palm"
JSON = './w1wr5ktymc_hole.geojsonl'
PIN = 'w1wr5ktymc'

gdfALRO = gpd.read_file(GDB[0], layer=GDB[1] )[
                ['geohash10','geometry']]
gdf = gdfALRO[gdfALRO.geohash10==PIN].copy()
gdf = gdf.explode( index_part=True)
gdf["geometry"] = gdf["geometry"].apply(drop_z)
gdf.rename(columns={"geohash10": "PIN"}, inplace=True)
gdf['part'] = 'outer'

#####################################################################
gdfHole = gpd.read_file("w1wr5ktymc_hole.geojsonl")
for col in gdf.columns:
    if col not in gdfHole.columns:
        gdfHole[col] = None   # fill missing columns
# match column order
gdfHole = gdfHole[gdf.columns]
gdfHole['part'] = 'hole'
gdf = pd.concat([gdf, gdfHole], ignore_index=True)
gdf = InsertPoly_with_Hole( gdf, 0, 1 )

############################################################
the_poly = gdf.loc[1, "geometry"]     # same Polygon
mrr, long_axis, perp_axis = mrr_axes(the_poly, perp_len=0.5)
row_mrr = gpd.GeoDataFrame(
    [{'PIN': 'w1wr5ktymc', 'geometry': mrr, 'part': 'MRR'}],
    geometry='geometry', crs=gdf.crs)

poly_hole = gdf[gdf.part=='outer_with_hole'].iloc[0].geometry
parts = split(poly_hole, perp_axis)
pieces = [geom for geom in parts.geoms if geom.geom_type == "Polygon"]
#######################################################################
print( f'Cutting doughnut , got {len(pieces)} pieces...' )
row_d1 = gpd.GeoDataFrame(
    [{'PIN': 'w1wr5ktymc', 'geometry': pieces[0], 'part': 'donut-1'}],
    geometry='geometry', crs=gdf.crs)
row_d2 = gpd.GeoDataFrame(
    [{'PIN': 'w1wr5ktymc', 'geometry': pieces[1], 'part': 'donut-2'}],
    geometry='geometry', crs=gdf.crs)

gdf = gpd.GeoDataFrame(pd.concat([gdf, row_mrr,row_d1,row_d2], ignore_index=True),
                       geometry='geometry', crs=gdf.crs)

gdf["n_hole"] = gdf["geometry"].apply( lambda g: len(g.interiors) if g.geom_type == "Polygon" else sum(len(p.interiors) for p in g.geoms) if g.geom_type == "MultiPolygon" else 0)
gdf["geom_type"] = gdf["geometry"].apply(lambda g: g.geom_type)
gdf['area_sqm'] = gdf['geometry'].area*(111_000*111_000)
gdf['area_sqm'] = (gdf['area_sqm'].round().astype('int64').map(lambda x: f"{x:,}"))

tab = gdf[['PIN','part','geom_type','n_hole','area_sqm']].copy()
tab.index = tab.index + 1
print( tab.to_markdown() )
for _, row in gdf.iterrows():
    print(f'{60*"="}')
    part = row['part']
    geom = row['geometry']
    wkt_str = wkt.dumps(geom, rounding_precision=6)  # คุมทศนิยม 6 ตำแหน่ง
    print(f"{part}: {wkt_str}")

DoPlot(gdf)

