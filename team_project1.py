import osmnx as ox

point = (-7.625, 111.192) # Koordinat Lawu

graph = ox.graph_from_point(point, dist=5000, network_type='walk')

ox.save_graph_geopackage(graph, filepath="jalur_lawu_lengkap.gpkg")
