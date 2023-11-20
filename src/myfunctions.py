import alphashape
import itertools
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import itertools

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.measure import label, find_contours, points_in_poly
from skimage.color import label2rgb
from shapely.ops import nearest_points
from descartes import PolygonPatch
#from shapely.geometry import Polygon

def convert_coordinates(eas,nor):
    """
    Convert coordinates from VL95 to WGS84 using swisstopo API. *Check if Geopandas can do this too*

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """
    api_url = 'http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting={}&northing={}&altitude=550.0&format=json'.format(nor,eas)
    response = requests.get(api_url)
    return(float(response.json()['easting']),float(response.json()['northing']))

def save_numpy_array(np_arr,SAVEFILE_NAME):
    """
    Save numpy array under specified name in '/data/processed' folder

    Parameters
    ----------
    np_arr : numpy.array
                numpy array to save
    SAVEFILE_NAME : string
                name for savefile

    Returns
    -------
    None
    """
    with open('../data/processed/{}.npy'.format(SAVEFILE_NAME), 'wb') as f:
        np.save(f, np_arr)
    return()

def load_numpy_array(SAVEFILE_NAME):
    """
    Load numpy array (with specified name) from '../data/processed' folder

    Parameters
    ----------
    SAVEFILE_NAME : string
                name of savefile to load

    Returns
    -------
    numpy.array
                numpy array (e.g. containing longitudes (float64) and latitudes (float64) in WGS84 coordinates)
    """
    with open('../data/processed/{}.npy'.format(SAVEFILE_NAME), 'rb') as f:
        return(np.load(f)) 

def convert_coordinates_multiple(koord_df,SAVEFILE_NAME):
    """
    Convert multiple coordinates from VL95 to WGS84 using swisstopo API. 
    If a savefile is found, data is loaded from savefile (fast). If no savefile is found data is computed using swisstopo API and is saved afterwards (slow).
    *Check if Geopandas can do this too*

    Parameters
    ----------
    koord_df : pandas.DataFrame
                Dataframe containing longitude and latitude coordinates in VL95 format
    SAVEFILE_NAME : string
                name of savefile to load (if present)

    Returns
    -------
    koord_df_wgs84 : numpy.array
                numpy array containing longitude and latitude coordinates in WGS84 format
    """
    if os.path.exists('../data/processed/{}.npy'.format(SAVEFILE_NAME)):
        print("Savefile found. Loading coordinates from savefile.")
        koord_df_wgs84 = load_numpy_array(SAVEFILE_NAME)

    else:
        koord_df_wgs84 = np.empty(koord_df.shape,dtype='float')
        for i in range(koord_df.shape[0]):

            api_url = 'http://geodesy.geo.admin.ch/reframe/lv95towgs84?easting={}&northing={}&altitude=550.0&format=json'.format(koord_df[i,1],koord_df[i,0])

            response = requests.get(api_url)
            koord_df_wgs84[i,0] = float(response.json()['easting'])
            koord_df_wgs84[i,1] = float(response.json()['northing'])

            save_numpy_array(koord_df_wgs84,SAVEFILE_NAME)
    return(koord_df_wgs84)

def plot_cluster_shapes(df_adresses,cluster_col):
    """
    Plot adresses on citymap with their according cluster

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """

    df_adresses = df_adresses[df_adresses[cluster_col].notna()]
    
    cluster_polygon_list = []
    for i,c in enumerate(df_adresses[cluster_col].unique()):
        cluster_adresses = df_adresses.loc[df_adresses[cluster_col]==c,['x','y']].values.tolist()
        cluster_alpha_shape = alphashape.alphashape(cluster_adresses, 120)
        # cluster_path = PolygonPatch(cluster_alpha_shape, alpha=0.2).get_path()
        # cluster_vertices = cluster_path.vertices
        # cluster_polygon = Polygon([(i[0], i[1]) for i in zip(cluster_vertices[:,0],cluster_vertices[:,1])])
        cluster_polygon_list.append(cluster_alpha_shape)

    gdf_cluster = gpd.GeoDataFrame(zip(df_adresses[cluster_col].unique(),cluster_polygon_list),columns=["cluster","geometry"])

    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=True)
    ax.set_facecolor('xkcd:white')
    colors = itertools.cycle(["tab:blue", "tab:orange", "tab:green","tab:red",'tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan'])
    for i,c in gdf_cluster.iterrows():  
        cluster_color=next(colors)
        cluster_adresses_x = df_adresses.loc[df_adresses[cluster_col]==c['cluster'],'x'].values.tolist()
        cluster_adresses_y = df_adresses.loc[df_adresses[cluster_col]==c['cluster'],'y'].values.tolist()
        ax.scatter(x=cluster_adresses_x, y=cluster_adresses_y, c=cluster_color, marker='.', s=5, zorder=3)
        ax.add_patch(PolygonPatch(c['geometry'], alpha=0.2,color=cluster_color))

    ax.set_title(cluster_col)
    return(fig)

def plot_cluster(df_adresses,cluster_col):
    """
    Plot adresses on citymap with their according cluster

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """

    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=True)
    ax.set_facecolor('xkcd:white')
    ax.scatter(x=df_adresses['x'], y=df_adresses['y'], c=df_adresses[cluster_col], marker='.', s=5, zorder=3)
    ax.set_title(cluster_col)
    return(fig)

def plot_cluster2(df_adresses,cluster_col):
    """
    Plot adresses on citymap with their according cluster

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """
    
    fig, ax = ox.plot_graph(G, node_color='gray', node_size=0, show=False)
    ax.set_facecolor('xkcd:white')
    colors = itertools.cycle(["tab:blue", "tab:orange", "tab:green","tab:red",'tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan'])

    grouped = df_adresses.groupby(cluster_col)
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', color=next(colors), marker='.', s=5, zorder=3)  #colors[key], label=key,

    ax.set_title(cluster_col)
    #ax.get_legend().remove()
    #fig.show()
    return()

def k_core(G, k):
    """
    Convert coordinates from VL95 to WGS84 using swisstopo API. *Check if Geopandas can do this too*

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """
    H = nx.Graph(G, as_view=True)
    H.remove_edges_from(nx.selfloop_edges(H))
    core_nodes = nx.k_core(H, k)
    H = H.subgraph(core_nodes)
    return G.subgraph(core_nodes)


def plot2img(fig):
    """
    Convert coordinates from VL95 to WGS84 using swisstopo API. *Check if Geopandas can do this too*

    Parameters
    ----------
    eas : float64
                longitude in VL95 coordinates
    nor : float64
                latitude in VL95 coordinates

    Returns
    -------
    tuple
                A tuple containing longitude (float64) and latitude (float64) in WGS84 coordinates
    """
    # remove margins
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # convert to image
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    as_rgba = np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))
    return as_rgba[:,:,:3]

def find_nearest_point(x):
    unary_union = gdf_cityblocks.drop(gdf_cityblocks.iloc[[x]].index).unary_union
    np = nearest_points(gdf_cityblocks.iloc[x]['centroid'],unary_union)[1]
    return(np)

def crop_cityblock_using_neighbors(i):
    unary_union = gdf_cityblocks_buffered.drop(i).unary_union
    cropped_cityblock = gdf_cityblocks_buffered.loc[i,'geometry'].difference(unary_union)
    gdf_cityblocks_buffered.loc[i,'geometry'] = cropped_cityblock
    return()