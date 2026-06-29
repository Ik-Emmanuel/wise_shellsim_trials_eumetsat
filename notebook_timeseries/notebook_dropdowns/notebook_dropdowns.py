"""
Notebook dropbown functions.

This notebook contains variables and functions that allow users to select areas using a combination of dropdowns and interactive maps.
It was developed as part of the Living Wales project.

Authors: Abigail Sanders, Dan Clewley, Emmanuel Nwokocha.

"""

import glob
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import ipywidgets as widgets
import ipyleaflet
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import threading
from shapely.geometry import shape, mapping, Point, Polygon
import math
from ipywidgets import RadioButtons, BoundedFloatText, Layout, IntProgress, VBox, HBox, HTML, Button
from ipyleaflet import Map, GeoData, LayersControl, FullScreenControl, DrawControl, basemaps
from ipyleaflet import TileLayer, Marker, Popup, LayerGroup, CircleMarker, GeoJSON

from shapely.ops import transform
import pyproj

# used as datetime library for time functions
import time


   # import threading
   #  import time
   #  import math
   #  from ipywidgets import HTML, VBox, IntProgress, Layout, display


# folders to look for shape files
SHAPEFILES_AREAS_FOLDER = "/home/jovyan/work/ShellSIM_Trials/notebook_timeseries/notebook_dropdowns/shapefiles"
# UPLOADS_FOLDER = "/home/jovyan/work/ShellSIM_Trials/notebook_timeseries/notebook_dropdowns/shapefiles/uploads"

# default shapefile search glob
vector_types_list = glob.glob(f"{SHAPEFILES_AREAS_FOLDER}/*")
vector_types_dict = {}

# Add in user uploads directory
vector_types_dict["1. DRAW AREA"] = SHAPEFILES_AREAS_FOLDER
# vector_types_dict["2. USER UPLOADS"] = UPLOADS_FOLDER

vector_types_dict = vector_types_dict | {
    os.path.basename(vector_type).replace("_", " "): vector_type
    for vector_type in vector_types_list
    if os.path.isdir(vector_type)
}


# declare accessible global variables to store objects reused across components 
global gpd_df  ## the geopandas dataframe for the selected shapefile
global col_name_var   ## suitable column name for site names within a shapefile
global AREA_SELECTION
global selected_polygon
global RESULTS
global get_polygon
global buffer_distance
global confirmed_buffer_distance



# instantiate variables
RESULTS = {}
RESULTS["global_selected_polygon"] = None
RESULTS["global_selected_area"] = None  # for drawn area from map
RESULTS["final_area_selection"] = None  # Setting the final selection 

RESULTS["global_selected_polygon_geomvalue"] = None  ## the geometry value of selected polygon
RESULTS["global_area_selection_type"] = None  # options are 1. Draw: if selection method is to draw on map  2. Select: if selection method is to select from map or shp file.

RESULTS["global_selected_polygon_type"] = None  # options are All: if all is selected and Selected: if a single one is selected
RESULTS["get_polygon"] = None
RESULTS["area_selection_type"] = None
RESULTS["buffer_distance"] = 100
RESULTS["start_date"] = None  # widget holding the selected start date
RESULTS["end_date"] = None    # widget holding the selected end date
AREA_SELECTION = None
selected_polygon = None



def helper():
    """This method will display all the available method and usage snippet to the user """
    html = widgets.HTML()
    docs = """
    <p> <b style='color:black'> List of available commands for selecting geographic sites for analysis </b> </p>

    """
    html.value = docs
    display(html)
    return None

# ================================= Helper Functions =========================================================
def set_global_result(key, value, results_dict):
    """ This function sets value to
    globally defined RESULTS dictionary """
    results_dict[key] = value
    
    
def get_global_result(key, results_dict):
    """ This reads value from globally defined RESULTS dictionary """
    # return RESULTS.get(key, "Nothing selected")
    return results_dict.get(key, None)


def mapper_preprocessor(geopandas_dataframe):
    """
    Prepares the vector geopandas dataframe ready for mapping.
    """
    # Make a copy if the DataFrame might be a slice
    geopandas_dataframe = geopandas_dataframe.copy()
    
    # Set the GeoDataFrame to geographic CRS for plotting
    geopandas_dataframe = geopandas_dataframe.to_crs(epsg=4326)
    return geopandas_dataframe


def convert_to_geojson(selected_polygon):
    """Given a geopandas dataframe of single site, else it takes just first row  
    this converts and return a geojson format of it """
    if isinstance(selected_polygon, gpd.GeoDataFrame):
        # geometry = selected_polygon.loc[0, 'geometry']
        geometry = selected_polygon.iloc[0]['geometry']
        # Convert to GeoJSON-like dictionary
        area_to_geojson = geometry.__geo_interface__
        return area_to_geojson
    else:
        print("Error converting: Area is not a geopandas dataframe")
        return None
    
def convert_to_geopandas_df(selected_polygon):
    """Given a geojson of a single site, 
    this converts and return a geopandas dataframe with one column = "geometry"  """
    # If selected_polygon is a dictionary representing a geometry
    if isinstance(selected_polygon, dict) and 'type' in selected_polygon and 'coordinates' in selected_polygon:
        # Convert dictionary to a GeoPandas DataFrame
        geom = shape(selected_polygon)
        area_gdf = gpd.GeoDataFrame({'geometry': [geom]})
        return area_gdf
    else:
        print("Error converting: Area is not in GeoJson format")
        return None

def convert_timestamps_to_strings(df):
    """
    Converts all Timestamp columns in the DataFrame to strings.
    """
    for col in df.columns:
        if isinstance(df[col].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) or df[col].dtype == 'datetime64[ns]' or df[col].dtype == 'datetime64[ms]':
            df[col] = df[col].astype(str)
    return df


def prepare_area_for_buffer():
    """ For some reason, before map click areas can have 
    buffers applied to them, reprojections are needed not sure why """
    
    #fetch the area selection type
    global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)
    
    #fetch the globally selected polygon 
    area_gdf_to_prepare = polygon_selected()
    
    # if its not a geopandas data frame return value as is 
    if not isinstance(area_gdf_to_prepare, gpd.GeoDataFrame):
        return area_gdf_to_prepare 
    
    # else check its not and empty value
    if not area_gdf_to_prepare.empty:
        # if its an area that was clicked and selected on the MAP
        if global_area_selection_type and global_area_selection_type == "Select":
           
            # Ensure the GeoDataFrame is in a projected CRS for accurate area calculation
            gpd_df_first_conversion  = area_gdf_to_prepare.to_crs(epsg=3857)

            # Set the GeoDataFrame back to geographic CRS for plotting
            gpd_df_second_conversion = gpd_df_first_conversion.to_crs(epsg=4326)

            # Convert any Timestamps to strings
            gpd_df_final_conversion = convert_timestamps_to_strings(gpd_df_second_conversion)

            return gpd_df_final_conversion
        else:
            # For drawn polygon just return what is draw or do some other checks needed
            return area_gdf_to_prepare
    else:
        # case where we can't identify selected polygon 
        return None
    


# ==================================== End of helper functions =======================================


    
def polygon_selected():
    """ This function fetches and returns value of selected polygon if it exists """
    selected_global_polygon =  get_global_result("global_selected_polygon", RESULTS)
    selected_global_polygon_geomvalue =  get_global_result("global_selected_polygon_geomvalue", RESULTS)
    selected_global_polygon_type =  get_global_result("global_selected_polygon_type", RESULTS)
    global_area_selection_type =  get_global_result("global_area_selection_type", RESULTS)
    
    
    # if final selection has been set, return that
    global_final_area_selection =  get_global_result("final_area_selection", RESULTS)
    if global_final_area_selection is not None:
        return global_final_area_selection
    
    # get and retrun user drawn polygon from map 
    if global_area_selection_type and global_area_selection_type == "Draw":
        drawn_polygon = get_global_result("global_selected_area", RESULTS)
        return  drawn_polygon
    
    # get and return user selected polygon area(s)
    elif global_area_selection_type and global_area_selection_type == "Select":
        
        if selected_global_polygon_type and selected_global_polygon_type == "All":
             # return whole geodataframe selected  if all is selected
            get_polygon = get_global_result("get_polygon", RESULTS)
            if get_polygon and get_polygon.value is not None:
                gpd_df_sub = gpd_df[gpd_df[col_name_var] == get_polygon.value]
                return gpd_df_sub
            else:
                gpd_df_sub = gpd_df
                return gpd_df_sub

        elif selected_global_polygon and selected_global_polygon_type == "Selected":
            try:
          
                # Option1: fetch object using fid
                identifier = selected_global_polygon.get("fid", None)
                if identifier is not None:
                    # find and return selected polygon 
                    gpd_df_sub = gpd_df[gpd_df["fid"] == identifier]
                    return gpd_df_sub
                
            except Exception as e:
                print("error occured", e)

    # returns None set read drop down value
    if not selected_global_polygon:
        get_polygon_dropdown =  get_global_result("get_polygon", RESULTS)
        polygon_dropdown_value = get_polygon_dropdown.value
        if polygon_dropdown_value is not None:
            gpd_df_sub = gpd_df[gpd_df[col_name_var] == polygon_dropdown_value]
            return gpd_df_sub
        else:
            print("Polygon not set")
    print("No polygon currently selected. Run map_and_select_area(polygon_select), click/draw and confirm area on map")
    return None



def polygon_selected_tobbox():
    """
    Fetches the currently selected polygon and returns its bounding box.

    Works for both selection methods:
      - "Select": polygon_selected() returns a GeoDataFrame -> use total_bounds
      - "Draw":   polygon_selected() returns a GeoJSON-like dict -> use shape().bounds

    Returns:
        tuple: (min_lon, min_lat, max_lon, max_lat), or None if nothing selected.
    """
    selected = polygon_selected()

    if selected is None:
        print("No polygon currently selected. Run map_and_select_area(polygon_select), click/draw and confirm area on map")
        return None

    # GeoDataFrame (selected from map/shapefile, or a confirmed buffer)
    if isinstance(selected, gpd.GeoDataFrame):
        if selected.empty:
            print("No polygon currently selected.")
            return None
        # Ensure geographic CRS so the bbox is in lon/lat
        if selected.crs is not None and selected.crs.to_epsg() != 4326:
            selected = selected.to_crs(epsg=4326)
        bounds = selected.total_bounds  # (minx, miny, maxx, maxy)
        return tuple(bounds)

    # GeoJSON-like dict (drawn area / buffered drawn area)
    if isinstance(selected, dict) and 'type' in selected and 'coordinates' in selected:
        geometry = shape(selected)
        return geometry.bounds  # (minx, miny, maxx, maxy)

    print("Error: selected area is not in a recognised format (GeoDataFrame or GeoJSON).")
    return None


def date_selection():
    """
    Displays widgets to set a start date and an end date for analysis.

    Dates are picked with calendar widgets and stored globally (in RESULTS) so
    they can be retrieved later with date_selected(). A 'Reset' button clears
    the currently set dates.

    Returns:
        tuple: (start_date_picker, end_date_picker) widgets.
    """
    style = {'description_width': 'initial'}

    # Calendar pickers for start and end dates
    start_date_picker = widgets.DatePicker(
        description='Start date',
        disabled=False,
        layout=Layout(width='40%'),
        style=style
    )

    end_date_picker = widgets.DatePicker(
        description='End date',
        disabled=False,
        layout=Layout(width='40%'),
        style=style
    )

    # Output area to echo the currently selected dates back to the user
    feedback = widgets.HTML()

    def update_feedback(*args):
        start = start_date_picker.value.isoformat() if start_date_picker.value else "Not set"
        end = end_date_picker.value.isoformat() if end_date_picker.value else "Not set"
        feedback.value = (
            f"<b style='color:#1a2172'> Start date: </b> {start} "
            f"<b style='color:#1a2172'> &nbsp; End date: </b> {end}<br>"
        )

    start_date_picker.observe(update_feedback, "value")
    end_date_picker.observe(update_feedback, "value")

    # Reset button clears both dates
    def reset_dates(*args):
        start_date_picker.value = None
        end_date_picker.value = None
        update_feedback()

    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_dates)

    # Store widgets globally so date_selected() can read them later
    set_global_result("start_date", start_date_picker, RESULTS)
    set_global_result("end_date", end_date_picker, RESULTS)

    # Initial feedback
    update_feedback()

    # Display the widgets and reset button
    display(HTML("<b style='color:#1a2172'> Use the calendars below to set a start date and an end date for analysis, <span style='color:orange'> or type in format \"MM/DD/YYYY\" (Month/Day/Year) </span>. <p> Use the 'Reset' button to clear the selected dates. </b><br>"))
    display(start_date_picker)
    display(end_date_picker)
    display(reset_button)
    display(feedback)

    return start_date_picker, end_date_picker


def date_selected():
    """
    Fetches the currently selected start and end dates.

    Returns:
        tuple: (start_date, end_date) as strings in 'YYYY-MM-DD' format
               (e.g. '2021-05-01'), with None for any date not yet set.
    """
    start_date_picker = get_global_result("start_date", RESULTS)
    end_date_picker = get_global_result("end_date", RESULTS)

    if start_date_picker is None or end_date_picker is None:
        print("No dates set. Run date_select = notebook_dropdowns.date_selection() and pick dates first.")
        return None, None

    start_date = start_date_picker.value.isoformat() if start_date_picker.value else None
    end_date = end_date_picker.value.isoformat() if end_date_picker.value else None

    return start_date, end_date


def area_selection():
    """Function that displays options to select an area, shapefile and polygon"""
    # Path to Welsh Dataset repository
    shapefiles_dict = {}

    def update_shapefiles(*args):
        args        # List all shapefiles in the selected directory
        shapefiles_list = glob.glob(
            os.path.join(vector_types_dict[get_type.value], "*.shp")
        )
        shapefiles_dict.clear()
        shapefiles_dict.update(
            {
                os.path.basename(shapefile)
                .replace(".shp", "")
                .replace("_", " ")
                .lower(): shapefile
                for shapefile in shapefiles_list
            }
        )

        # Update shapefile dropdown options
        get_shapefile.options = list(shapefiles_dict.keys())
        get_shapefile.value = (
            list(shapefiles_dict.keys())[0] if shapefiles_dict else None
        )
        update_polygons()

    # Function to update the polygon options
    def update_polygons(*args):
        selected_shapefile_path = shapefiles_dict.get(get_shapefile.value, None)

        if selected_shapefile_path:
            global gpd_df, col_name_var  # Define as global variables
            gpd_df = gpd.read_file(selected_shapefile_path)
            ## Very important. it adds the unique identifier to be used to identifiy polygons with shp file
            gpd_df["fid"] = gpd_df.index

            # Try to find a suitable column name for site names
            col_name_var = None
            if "name" in gpd_df.columns:
                col_name_var = "name"
            else:
                for col in gpd_df.columns:
                    if "name" in col.lower():
                        col_name_var = col
                        break

            if col_name_var is not None:
                site_names = gpd_df[col_name_var].drop_duplicates().tolist()
                get_polygon.options = site_names
                get_polygon.value = None
            else:
                get_polygon.options = []
                get_polygon.value = None
        else:
            get_polygon.options = []
            get_polygon.value = None

    style = {'description_width': 'initial'}
    
    # Dropdown for selecting vector type
    get_type = widgets.Dropdown(
        options=list(vector_types_dict.keys()),
        value=list(vector_types_dict.keys())[0],
        default="User uploads",
        description="Area Selection Type",
        disabled=False,
        layout=Layout(width='40%'),
        style=style
    )

    # Dropdown for selecting shapefile
    get_shapefile = widgets.Dropdown(
        options=[],
        description="Filter dataset or select all",
        disabled=False,
        layout=Layout(width='40%'),
        style=style
    )

    # Dropdown for selecting polygon
  
    get_polygon = widgets.Dropdown(
        options=[],
        description="OPTIONAL - Select a site",
        disabled=False,
        default="",
        layout=Layout(width='40%'),
        style=style
    )

    # Observe changes and update accordingly
    get_type.observe(update_shapefiles, "value")
    get_shapefile.observe(update_polygons, "value")

    # Function to reset the dropdowns and clear outputs
    def reset_dropdowns(*args):
        get_type.value = list(vector_types_dict.keys())[0]
        update_shapefiles()
        clear_output(wait=True)
        display(get_type)
        display(get_shapefile)
        display(get_polygon)
        display(reset_button)

    # Button for resetting the dropdowns
    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_dropdowns)

    # Initial update of shapefiles and polygons
    update_shapefiles()

    # Display the dropdowns and reset button
    display(HTML("<b style='color:#1a2172'> Use the dropdown menu below to select areas or sites for analysis <p> Use the 'Reset' button to clear selection </b><br>"))
    display(get_type)
    display(get_shapefile)
    display(get_polygon)
    display(reset_button)
    # return get_type, get_shapefile, get_polygon, reset_button
    
    set_global_result("get_polygon", get_polygon, RESULTS)
    set_global_result("area_selection_type", get_type, RESULTS)
    return get_polygon



def view_selected_polygon(selected_polygon):
    """
    returns a geodataframe of the selected polygon 
    """
    if selected_polygon.value is not None:
        gpd_df_sub = gpd_df[gpd_df[col_name_var] == selected_polygon.value]
        polygon_name = selected_polygon.value
    else:
        gpd_df_sub = gpd_df
        polygon_name = "All"

    return gpd_df_sub


# formally called:  static_polygon_plot
def plot_selected_polygon(selected_polygon):
    """
    Produces a static plot of a given polygon
    """
    # ================ add progress bar =========
    progress_value = IntProgress(min=0, max=100) # instantiate the progress bar
    print("Generating Plot ...")
    display(progress_value) # display the bar
    
    stop_thread = threading.Event()  # Event used to signal the thread to stop
    
    def update_progress_bar():
        """Continuously update the progress bar until the map is ready."""
        progress = 0
        while not stop_thread.is_set():  # Continue until stop signal is received
            progress_value.value = progress % 100
            progress += 1
            time.sleep(0.2)
        progress_value.value = 100  # Ensure progress bar is set to 100% when stop_thread is set
        
    
     # Start progress bar in a separate thread
    progress_thread = threading.Thread(target=update_progress_bar)
    progress_thread.start()

    if selected_polygon.value is not None:
        gpd_df_sub = gpd_df[gpd_df[col_name_var] == selected_polygon.value]
        polygon_name = selected_polygon.value
    else:
        gpd_df_sub = gpd_df
        polygon_name = "All"

    # Ensure the GeoDataFrame is in a projected CRS for accurate area calculation
    gpd_df_sub = gpd_df_sub.to_crs(epsg=3857)

    # Calculate the area in square meters
    gpd_df_sub["area"] = gpd_df_sub.geometry.area

    # Sum the areas to get the total area in hectares (1 hectare = 10,000 square meters)
    total_area = gpd_df_sub["area"].sum() / 10000

    # Set the GeoDataFrame back to geographic CRS for plotting
    gpd_df_sub = gpd_df_sub.to_crs(epsg=4326)

    # Set the figure size for standardization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Visualize the polygon with standardized size
    gpd_df_sub.plot(ax=ax, color="blue", edgecolor="black")
    ax.set_title(f"Site Visualization ({polygon_name})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add north arrow
    x, y = -0.2, 1  # Adjust these values based on your plot
    arrow_length = 0.1
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="black", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=20,
        xycoords="axes fraction",
    )
    
    
    plt.show()
    # Display total area
    print(f"Total area: {total_area:.2f} ha")
    
        # Signal the progress thread to stop before function exists
    stop_thread.set()
    progress_thread.join()  # Ensure the progress thread has finished 

    return None




def map_and_select_area(selected_polygon):
    """" Function to map selected polygon and click to select or draw to select """
    # fetch geodataframe of selected polygon
    if selected_polygon.value is not None:
        gpd_df_sub = gpd_df[gpd_df[col_name_var] == selected_polygon.value]
        polygon_name = selected_polygon.value
    else:
        gpd_df_sub = gpd_df
        polygon_name = "All"
        
    gpd_df_sub = mapper_preprocessor(gpd_df_sub)
                
    # identify if DRAW ON MAP or others 
    area_selection_type = get_global_result("area_selection_type", RESULTS)
    if area_selection_type and area_selection_type.value:
        if area_selection_type.value.endswith("DRAW AREA"):
            # set global area selection type to: Draw
            set_global_result("global_area_selection_type", "Draw", RESULTS)
            # show map to draw area
            draw_site_from_map(gpd_df_sub)
        else: 
            # set global area selection type to: Select
            set_global_result("global_area_selection_type", "Select", RESULTS)
            # show interactive map to select site from 
            select_site_from_map(gpd_df_sub)

    else:
        print("Unidentified Area Selection Type")
        return None
    
    

def select_site_from_map(gpd_df_sub):
    """
    Produces an interactive plot of a given polygon for click and select
    """

    stop_thread = threading.Event()  # Event to signal the thread to stop
    # Initialize selected_polygon variable
    selected_polygon = None

    # Style dictionary for non-selected polygons
    default_style = {
        "color": "black",
        "fillColor": "#3366cc",
        "opacity": 0.05,
        "weight": 1.9,
        "dashArray": "2",
        "fillOpacity": 0.6,
    }

    # Style dictionary for the selected polygon
    selected_style = {
        "color": "black",
        "fillColor": "orange",  # Color for the selected polygon
        "opacity": 0.8,
        "weight": 2,
        "dashArray": "2",
        "fillOpacity": 0.6,
    }

    
    # Function to update the style of the polygons
    def update_polygon_style():
        for feature in geo_data.data['features']:
            if selected_polygon and all(
                feature["properties"][k] == selected_polygon[k] for k in selected_polygon
            ):
                feature['style'] = selected_style
            else:
                feature['style'] = selected_style
        
    
    # Function to confirm and rename the output to AREA_selection
    def confirm_selection(button):
        AREA_SELECTION = AREA_SELECTION
        print("The selected area has been confirmed as 'AREA_SELECTION'")

    # Function to select all polygons 
    def confirm_select_all(button):
        selected_polygon = None  # Clear any individual polygon  selection
        update_polygon_style()  # Reset the styles
        html.value = "<b style='color:orange'>  All polygons currently selected <b><br>"
        set_global_result("global_selected_polygon", gpd_df_sub, RESULTS)
        # set final_area_selection from applied buffer if any, to none
        set_global_result("final_area_selection", None, RESULTS)
        set_global_result("global_selected_polygon_type", "All", RESULTS)
     

    # ================ add progress bar =========
    progress_value = IntProgress(min=0, max=100) # instantiate the bar
    print("Generating Interactive Map ...")
    display(progress_value) # display the progress  bar as a widget

    def update_progress_bar():
        """Continuously update the progress bar until the map is ready."""
        progress = 0
        while not stop_thread.is_set():  # Continue until stop signal is received
            progress_value.value = progress % 100
            progress += 1
            time.sleep(0.2)
        progress_value.value = 100  #
    
    # Start progress bar in a separate thread
    progress_thread = threading.Thread(target=update_progress_bar)
    progress_thread.start()

    # Calculate the bounding box
    bounds = gpd_df_sub.total_bounds  # returns (minx, miny, maxx, maxy)
    sw = [bounds[1], bounds[0]]  # southwest corner (miny, maxx)
    ne = [bounds[3], bounds[2]]  # northeast corner (maxy, minx)

    # Calculate the center of the bounding box
    center = [(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2]

    # Convert any Timestamps to strings
    AREA_SELECTION = convert_timestamps_to_strings(gpd_df_sub)

    # Initialize selected_polygon variable
    selected_polygon = None

    # Create a button for confirming the selection
    confirm_button = widgets.Button(description="CONFIRM")
    confirm_button.on_click(confirm_selection)
    

    html = widgets.HTML()
    html.value = "<b style='color:orange'> All polygons currently selected <b>"
    
    #use all polygon button
    select_all_poly_button = widgets.Button(description="USE ALL POLYGONS")
    select_all_poly_button.on_click(confirm_select_all)

    # Create GeoData layer
    geo_data = ipyleaflet.GeoData(
        geo_dataframe=AREA_SELECTION,
        style=default_style,
        hover_style={"fillColor": "red", "fillOpacity": 0.2},
        name="Boundary",
    )
    
    # # Create GeoData layer
    # selected_data = ipyleaflet.GeoData(
    #     geo_dataframe=AREA_SELECTION,
    #     style=selected_style,
    #     hover_style={"fillColor": "orange", "fillOpacity": 0.2},
    #     name="Selected",
    # )
    
    # Function to handle click events and store the selected polygon
    def handle_click(event, feature, **kwargs):
        html.value = f"<b style='color:orange'> Identifying selected area wait .... </b> <br><br>"
        global selected_polygon
        selected_polygon = feature["properties"]
        selected_polygon_geomvalue = feature["geometry"]
        # Update the style of the selected polygon
        update_polygon_style()
        html.value = f"<b style='color:orange'> Selected Polygon: </b> <br> <b style='color:#1a2172'> {selected_polygon} </b> <br>"
        set_global_result("global_selected_polygon", selected_polygon, RESULTS)
        # set final_area_selection from applied buffer if any, to none
        set_global_result("final_area_selection", None, RESULTS)
        set_global_result("global_selected_polygon_geomvalue", selected_polygon_geomvalue, RESULTS)
        set_global_result("global_selected_polygon_type", "Selected", RESULTS)
        
        
    geo_data.on_click(handle_click)

    # Create a map centered on the GeoDataFrame
    m = ipyleaflet.Map(
        center=center,
        zoom=50,
        basemap=ipyleaflet.basemaps.Esri.WorldImagery,
        layout=widgets.Layout(height="600px"),
    )

    # Add GeoData layer to the map

    # m.add_layer(selected_data)
    m.add_layer(geo_data)
  
    # Fit map to bounds
    m.fit_bounds([sw, ne])

    # Add controls to the map
    m.add_control(ipyleaflet.LayersControl(position="topright"))
    m.add_control(ipyleaflet.FullScreenControl())

    # Display the map and UI elements
    display(HTML("<b style='color:#1a2172'> Follow instructions below to correctly select area of interest. </b><br>")) 
    display(
        widgets.VBox(
            [
                widgets.HTML(
                    "<b>To use all areas shown on the map, click <span style='color:orange'> 'USE ALL POLYGONS' </span>.<br> If you want to select a specific polygon click on the map, to select area and <span style='color:orange'> wait for <span style='color:#5a5c5a'> 'Selected Polygon' </span> confirmation below.<span>  </b>"
                ),
                html,
                select_all_poly_button,
                m,
            ]
        )
    )
   


    # Stop progress thread
    stop_thread.set()
    # Ensure progress thread has finished before exiting function
    progress_thread.join()




    
# ==================  Draw site from map 


# python
def draw_site_from_map(gpd_df_sub):
 

    stop_thread = threading.Event()  # Event to signal the thread to stop

    # ================ helper: convert circle drawn as point to polygon =========
    def point_to_circle(center, radius_meters):
        """Convert a point and radius (in meters) to a polygon approximating a circle"""
        earth_radius = 6371000  # Earth's radius in meters
        angular_radius = radius_meters / earth_radius  # Convert to angular radius

        coords = []
        for i in range(64):  # 64 points to approximate a circle
            angle = i * (2 * math.pi / 64)
            lat = math.asin(
                math.sin(center[1] * math.pi / 180) * math.cos(angular_radius) +
                math.cos(center[1] * math.pi / 180) * math.sin(angular_radius) * math.cos(angle)
            )
            lon = center[0] * math.pi / 180 + math.atan2(
                math.sin(angle) * math.sin(angular_radius) * math.cos(center[1] * math.pi / 180),
                math.cos(angular_radius) - math.sin(center[1] * math.pi / 180) * math.sin(lat)
            )
            coords.append((lon * 180 / math.pi, lat * 180 / math.pi))
        coords.append(coords[0])  # Close the polygon
        return {"type": "Polygon", "coordinates": [coords]}

    # ================ UI: progress bar =========
    progress_value = IntProgress(min=0, max=100)  # instantiate the bar
    print("Generating Interactive Map ...")
    display(progress_value)  # display the progress bar as a widget

    def update_progress_bar():
        """Continuously update the progress bar until the map is ready."""
        progress = 0
        while not stop_thread.is_set():  # Continue until stop signal is received
            progress_value.value = progress % 100
            progress += 1
            time.sleep(0.2)
        progress_value.value = 100

    progress_thread = threading.Thread(target=update_progress_bar)
    progress_thread.start()

    boundary = gpd_df_sub

    # HTML widget to display selected area information
    html = HTML()
    html.value = "<b> Draw an area on the map below, using the left-hand toolbar, within the highlighted boundary.</b>"

    # Create GeoData layer for the boundary
    geo_data = GeoData(
        geo_dataframe=boundary,
        style={'color': 'red', 'fillColor': 'none', 'opacity': 1, 'weight': 2},
        name='Boundary'
    )

    # Create a map centered on the boundary with appropriate zoom
    bounds = boundary.total_bounds  # returns (minx, miny, maxx, maxy)
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = Map(center=center, zoom=3, basemap=basemaps.Esri.WorldImagery,
            layout=Layout(height='600px'), scroll_wheel_zoom=True)

    # Overlay: City/country labels for all zoom levels
    city_layer = TileLayer(
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        name="White Area Labels",
        attribution="Tiles © Esri",
        opacity=1.0
    )
    m.add_layer(city_layer)

    # Add GeoData layer to the map
    m.add_layer(geo_data)

    # Marker layer to hold markers placed via the Draw-a-marker tool
    marker_layer = LayerGroup(name='Markers')
    m.add_layer(marker_layer)

    # Layer that holds the single rendered selection (polygon / rectangle /
    # line / circle). Markers live in marker_layer above. Only ONE selection
    # of any kind is ever kept at a time.
    selection_layer = LayerGroup(name='Selection')
    m.add_layer(selection_layer)

    # Guard flag so that clearing the draw control programmatically (which can
    # itself fire a 'deleted' draw event in some ipyleaflet versions) does not
    # re-enter handle_draw and wipe the selection we are in the middle of making.
    _state = {"clearing": False}

    # Render a GeoJSON geometry (Polygon / LineString / Rectangle) as the single
    # visible selection on the map.
    def render_selection_geometry(geom):
        try:
            feature = {"type": "Feature", "properties": {}, "geometry": geom}
            gj = GeoJSON(
                data=feature,
                style={'color': '#ff0000', 'weight': 4,
                       'fillColor': '#ff0000', 'fillOpacity': 0.2},
            )
            selection_layer.add_layer(gj)
        except Exception:
            pass

    # Helper to add a marker with hover tooltip and popup showing lat/lon
    def add_marker(lat, lon, make_select=False):
        nonlocal selected_area
        # Create marker at clicked location
        marker = Marker(location=(lat, lon), draggable=False)

        # Popup with lat/lon display
        try:
            popup_html = HTML(value=f"<b>Lat:</b> {lat:.6f} <br> <b>Lon:</b> {lon:.6f}")
            popup = Popup(location=(lat, lon), child=popup_html, close_button=False, auto_close=False)
        except Exception:
            # Fallback: attach a simple HTML to marker.popup if Popup init signature differs
            popup_html = HTML(value=f"<b>Lat:</b> {lat:.6f} <br> <b>Lon:</b> {lon:.6f}")
            popup = popup_html

        # Hover tooltip showing lat/lon. ipyleaflet Marker exposes a `title`
        # trait that renders as a native tooltip when hovering over the pin.
        try:
            marker.title = f"Lat: {lat:.6f}, Lon: {lon:.6f}"
        except Exception:
            pass

        # Click: open popup (add popup layer to map)
        def on_marker_click(**kwargs):
            try:
                # If popup is a Popup instance, add it to the map
                if isinstance(popup, Popup):
                    # ensure only one popup instance on the map for this marker
                    if popup not in m.layers:
                        m.add_layer(popup)
                else:
                    # if popup is a widget, create a Popup wrapper and add
                    popup_wrap = Popup(location=(lat, lon), child=popup_html, close_button=False, auto_close=False)
                    m.add_layer(popup_wrap)
            except Exception:
                pass

        try:
            marker.on_click(on_marker_click)
        except Exception:
            # on_click might not exist in some versions; fallback is to add the popup directly
            pass

        # Try to open popup on mouseover / close on mouseout if supported
        try:
            def _mouseover(**kw):
                try:
                    if isinstance(popup, Popup) and popup not in m.layers:
                        m.add_layer(popup)
                except Exception:
                    pass

            def _mouseout(**kw):
                try:
                    if isinstance(popup, Popup) and popup in m.layers:
                        m.remove_layer(popup)
                except Exception:
                    pass

            marker.on_mouseover(_mouseover)
            marker.on_mouseout(_mouseout)
        except Exception:
            # ignore if these events are not available
            pass

        # Add marker to the marker layer
        marker_layer.add_layer(marker)

        # If this marker should be treated as a selected area, store it
        if make_select:
            selected_area = {"type": "Point", "coordinates": [lon, lat]}
            html.value = f"<b> <span style='color:orange'> Selected Area: </span> <br> <b style='color:#1a2172'> {selected_area} </b></b>"
            try:
                set_global_result("global_selected_area", selected_area, RESULTS)
                set_global_result("final_area_selection", None, RESULTS)
            except Exception:
                pass

    # Add drawing tools to the map
    draw_control = DrawControl(
        polygon={"shapeOptions": {"color": "#ff0000", "weight": 4}},
        polyline={"shapeOptions": {"color": "#ff0000", "weight": 4}},
        circle={"shapeOptions": {"color": "#ff0000", "weight": 4}},
        rectangle={"shapeOptions": {"color": "#ff0000", "weight": 4}},
        marker={"shapeOptions": {"color": "#ff0000", "weight": 4}},
        circlemarker={},
    )

    # Helper to clear a LayerGroup across ipyleaflet versions
    def _clear_layer_group(group):
        try:
            group.clear_layers()
        except Exception:
            # Fallback for ipyleaflet versions without clear_layers()
            for lyr in list(group.layers):
                try:
                    group.remove_layer(lyr)
                except Exception:
                    pass

    # Remove the draw control's own rendered shapes. Guarded so the resulting
    # 'deleted' event (fired by some ipyleaflet versions) doesn't re-enter
    # handle_draw. We deliberately do NOT touch draw_control.data here: setting
    # it to [] makes the frontend echo an empty-data update and forces a map
    # bounds re-sync, which both wiped the selection and raised the
    # "'east' trait expected a float, not None" TraitError.
    def clear_draw_control():
        _state["clearing"] = True
        try:
            draw_control.clear()
        except Exception:
            pass
        finally:
            _state["clearing"] = False

    # Clear EVERYTHING: dropped marker pin, rendered selection, the draw
    # control's own shapes, and reset the stored selection. Enforces the
    # "only one selection at a time" rule.
    def clear_selection():
        nonlocal selected_area
        _clear_layer_group(marker_layer)
        _clear_layer_group(selection_layer)
        clear_draw_control()
        selected_area = None
        html.value = "<b> Draw an area on the map below, using the left-hand toolbar, within the highlighted boundary.</b>"
        try:
            set_global_result("global_selected_area", None, RESULTS)
            set_global_result("final_area_selection", None, RESULTS)
        except Exception:
            pass

    # Store a polygon/line/circle geometry as the single selection and render it
    def set_area_geometry(geom):
        nonlocal selected_area
        selected_area = geom
        render_selection_geometry(geom)
        html.value = f"<b> <span style='color:orange'> Selected Area: </span> <br> <b style='color:#1a2172'> {selected_area} </b></b>"
        try:
            set_global_result("global_selected_area", selected_area, RESULTS)
            set_global_result("final_area_selection", None, RESULTS)
        except Exception:
            pass

    # Handle drawings made from the toolbar (rectangle, polygon, line, circle,
    # marker). A selection is ONLY ever made through these tools — a plain click
    # on the map does nothing. Every new drawing clears the previous one so that
    # exactly one area (or one marker point) exists at any time.
    def handle_draw(self, action, geo_json):
        nonlocal selected_area
        # Ignore events fired by our own programmatic clears (clear() can emit a
        # 'deleted' event in some ipyleaflet versions).
        if _state["clearing"]:
            return
        # The trash / "Clear All" button fires a 'deleted' action.
        if action == 'deleted':
            clear_selection()
            return
        # Only react to a freshly created shape.
        if action != 'created':
            return

        # Remove any previous selection (old shape, marker, and the draw
        # control's native shapes — including the one just drawn, which we
        # re-render ourselves below). This enforces a single selection without
        # touching draw_control.data, so nothing gets wiped by a frontend echo.
        _clear_layer_group(marker_layer)
        _clear_layer_group(selection_layer)
        clear_draw_control()

        geometry = geo_json.get('geometry', {})
        gtype = geometry.get('type')
        props = geo_json.get('properties', {}) or {}
        style = props.get('style', {}) or {}

        if gtype == 'Point' and 'radius' not in style and props.get('shape') != 'circle':
            # A marker placed with the Draw-a-marker tool: single point only.
            lon, lat = geometry['coordinates'][0], geometry['coordinates'][1]
            add_marker(lat, lon, make_select=True)
        elif gtype == 'GeometryCollection' and props.get('shape') == 'circle':
            # Circle represented as a GeometryCollection with a radius.
            try:
                center = geometry['geometries'][0]['coordinates']
                radius = style.get('radius', None)
                if radius is not None:
                    set_area_geometry(point_to_circle(center, radius))
            except Exception:
                pass
        elif gtype == 'Point' and 'radius' in style:
            # Circle represented as a point + radius.
            set_area_geometry(point_to_circle(geometry['coordinates'], style['radius']))
        else:
            # Polygons, rectangles, lines.
            set_area_geometry(geometry)

    draw_control.on_draw(handle_draw)

    m.add_control(draw_control)

    # NOTE: a plain click on the map intentionally does NOT create a point.
    # Points are only added via the Draw-a-marker tool in the toolbar.

    # Add controls to the map
    m.add_control(LayersControl(position='topright'))
    m.add_control(FullScreenControl())

    # Display the initial map with the HTML widget
    display(HTML("<b style='color:#1a2172'> After drawing area of interest on the map, wait for 'Selected Area:' confirmation below, before proceeding </b><br>"))
    display(VBox([html, m]))

    # Initialize selected_area variable
    selected_area = None

    # Stop progress thread now that map is displayed and handlers are attached
    stop_thread.set()
    progress_thread.join()
    # function ends here; selected_area/global results have been set when user interacts    
    

    
    


    
# ========================= Visualize selection  ======================== 


# Function to display AREA_selection on a map if in geopandas df format
def display_geopandas_df_selection(area_selection, outline_color="black"):
    """ Given a df this allows to map area for visual confirmation """
    # Explicitly create a copy if needed
    area_selection = area_selection.copy()
    
    
    area_selection = convert_timestamps_to_strings(area_selection)
    area_selection = mapper_preprocessor(area_selection)
    if not area_selection.empty:
        # Create GeoData layer for AREA_selection
        selection_geo_data = GeoData(
            geo_dataframe=area_selection,
            style={
                "color": outline_color,
                "fillColor": "#3366cc",
                "opacity": 1,
                "weight": 3,
                # "dashArray": "2",
                "fillOpacity": 0.6,
            },
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            name="AREA Selection",
        )
        
        # Calculate the center of the selection area
        bounds = area_selection.total_bounds  # returns (minx, miny, maxx, maxy)
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create a map centered on the selection area
        selection_map = Map(center=center, zoom=10, basemap=basemaps.Esri.WorldImagery, layout=Layout(height='600px'))
        
        # Add GeoData layer to the map
        selection_map.add_layer(selection_geo_data)
        
        # Fit map to bounds
        sw = [bounds[1], bounds[0]]  # southwest corner (miny, minx)
        ne = [bounds[3], bounds[2]]  # northeast corner (maxy, maxx)
        selection_map.fit_bounds([sw, ne])
        
        # Add controls to the map
        selection_map.add_control(LayersControl(position='topright'))
        selection_map.add_control(FullScreenControl())
        
        # Display the map
        display(HTML("<b style='color:#1a2172'> Check that area displayed is your area of interest before proceeding, if not, reselect area above. <p> If this area is correct proceed to add a buffer (section 1.2), if required, or move to section 2.0. </b><br>"))      
        display(selection_map)
    else:
        display(HTML("No area selected."))


# Function to visualize the selected area on a new map if in GeoJson format
def visualize_selected_area():
    """ This function visualizes drawn selected area that is in GeoJson format  """
    selected_global_polygon =  get_global_result("global_selected_polygon", RESULTS)
    selected_global_polygon_geomvalue =  get_global_result("global_selected_polygon_geomvalue", RESULTS)
    global_area_selection_type =  get_global_result("global_area_selection_type", RESULTS)
    global_area_selection = get_global_result("global_selected_area", RESULTS)
    

    # get and retrun user drawn polygon from map 
    if global_area_selection_type and global_area_selection_type == "Draw":
        drawn_polygon = get_global_result("global_selected_area", RESULTS)

        if drawn_polygon:
            selected_area = drawn_polygon
            # Convert the selected area to a GeoDataFrame
            selected_geom = shape(selected_area)
            selected_gdf = gpd.GeoDataFrame({'geometry': [selected_geom]}, crs='epsg:4326')

            # If the selection is a single point, there is no area to render as a
            # polygon. Draw a red circle around the point and zoom out a bit so
            # the user can see where their selected point sits.
            geom_type = selected_area.get('type') if isinstance(selected_area, dict) else selected_geom.geom_type
            if geom_type == 'Point':
                lon, lat = selected_area['coordinates'][0], selected_area['coordinates'][1]

                # Map centered on the point, zoomed out a bit for context
                selected_map = Map(center=[lat, lon], zoom=8, basemap=basemaps.Esri.WorldImagery, layout=Layout(height='600px'))

                # Red circle marker highlighting the selected point (pixel radius,
                # so it stays visible at any zoom level)
                point_circle = CircleMarker(
                    location=(lat, lon),
                    radius=15,
                    color="red",
                    fill_color="red",
                    fill_opacity=0.3,
                    weight=3,
                    name="Selected Point",
                )
                selected_map.add_layer(point_circle)

                # Controls
                selected_map.add_control(LayersControl(position='topright'))
                selected_map.add_control(FullScreenControl())

                # Confirm button stores the point GeoDataFrame as AREA_selection
                def confirm_selection(button):
                    global AREA_selection
                    AREA_selection = selected_gdf
                    display(HTML("<b>The selected point has been confirmed as AREA_selection</b>"))

                confirm_button = Button(description="CONFIRM")
                confirm_button.on_click(confirm_selection)

                display(HTML("<b style='color:#1a2172'> Check that the highlighted point is your area of interest before proceeding. If not, reselect above. <p> If correct, proceed to add a buffer or to section 2.0. </b><br>"))
                display(HTML(f"<b style='color:#1a2172'> Selected point: Lat {lat:.6f}, Lon {lon:.6f} </b><br>"))
                display(VBox([HTML("<b></b>"), selected_map]))
                return None

            # Calculate the area in hectares
            proj = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True).transform
            selected_gdf['area_ha'] = selected_gdf['geometry'].apply(lambda geom: transform(proj, geom).area / 10000)
            area_ha = selected_gdf['area_ha'].iloc[0]

            # Transform shapefile boundaries into geographic data (and affect a style)
            geo_data = GeoData(
                geo_dataframe=selected_gdf,
                style={
                    "color": "red",
                    "fillColor": "#3366cc",
                    "opacity": 1,
                    "weight": 3,
                    # "dashArray": "2",
                    "fillOpacity": 0.6,
                },
                hover_style={"fillColor": "red", "fillOpacity": 0.2},
                name="Selected Area",
            )

            # Calculate the center of the selected area
            bounds = selected_gdf.total_bounds  # returns (minx, miny, maxx, maxy)
            center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

            # Create a map centered on the selected area
            selected_map = Map(center=center, zoom=10, basemap=basemaps.Esri.WorldImagery, layout=Layout(height='600px'))

            # Add GeoData layer to the map
            selected_map.add_layer(geo_data)

            # Fit map to bounds
            sw = [bounds[1], bounds[0]]  # southwest corner (miny, minx)
            ne = [bounds[3], bounds[2]]  # northeast corner (maxy, maxx)
            selected_map.fit_bounds([sw, ne])

            # Add controls to the map
            selected_map.add_control(LayersControl(position='topright'))
            selected_map.add_control(FullScreenControl())

            # Function to confirm and rename the output to AREA_selection
            def confirm_selection(button):
                global AREA_selection
                AREA_selection = selected_gdf
                display(HTML("<b>The selected area has been confirmed as AREA_selection</b>"))

            # Create a button for confirming the selection
            confirm_button = Button(description="CONFIRM")
            confirm_button.on_click(confirm_selection)

            # Display the map, button, and area in hectares
            
            display(HTML("<b style='color:#1a2172'> Check that area displayed is your area of interest before proceeding. If not, reselect area above </b>"))
            display(HTML(f"<b style='color:#1a2172'> Selected area: {area_ha:.2f} hectares </b><br>"))
            display(VBox([HTML("<b></b>"), selected_map, HTML(f"Area: {area_ha:.2f} hectares")]))
        else:
           
            display(HTML("<b style='color:#1a2172'> No area selected. </b><br>"))
    else: 
        # display(HTML("<b> <span style='color:orange'> No visuals </span>: Selected area was not drawn from map. Below is the currently selected area details</b>"))
        selected_area = polygon_selected()
        # selected_area = get_global_result("global_selected_area", RESULTS)
        
        if selected_area is not None and not selected_area.empty:
            display_geopandas_df_selection(selected_area, outline_color="red")
        else:
            display(HTML("No area selected."))
        return None
    
# # Example: Visualize the selected area stored in 'selected_area' from PART 2
# visualize_selected_area(selected_area)



# ================================= Add a buffer including your site ========================================== 

# Global variable for buffer distance
buffer_distance = 100


# Radio buttons widget for selecting buffer distance
buffer_distance_options = RadioButtons(
    options=[('100m', 100), ('500m', 500), ('1km', 1000), ('5km', 5000), ('10km', 10000), ('25km', 25000), ('50km', 50000), ('CUSTOM', 'CUSTOM')],
    value=100,
    description='Buffer Distance:',
    layout=Layout(width='300px')  # Set the width of the widget
)


def active_buffer():
    selected_buffer = get_global_result("buffer_distance", RESULTS)
    if selected_buffer:
        in_km = int(selected_buffer)/1000
        in_meters = int(selected_buffer)
        buffer_to_display = f"{in_km} km ({in_meters} meters)"
    else:
        buffer_to_display = "Not Selected"
        
    html = widgets.HTML()
    docs = f""" <p style='color:black'>  Buffer distance:  <b style='color:'>  {buffer_to_display} </b>  </p>"""
    html.value = docs
    display(html)
    

# Function to handle custom buffer distance input
def on_custom_buffer_distance_change(change):
    buffer_distance = change['new'] * 1000  # Convert km to meters
    set_global_result("buffer_distance", buffer_distance, RESULTS)


# Function to handle buffer distance selection
def on_buffer_distance_change(change):
    if change['new'] == 'CUSTOM':
        custom_buffer_distance.layout.display = ''  # Show custom distance input
    else:
        custom_buffer_distance.layout.display = 'none'  # Hide custom distance input
        if change.get("new"):
            buffer_distance = change.get("new")
            set_global_result("buffer_distance", buffer_distance, RESULTS) 


# Function to create buffer and display it on a map
def create_and_display_buffer_include_selection(area_gdf, buffer_distance):
    
    # prepare the area for buffer application
    area_gdf = prepare_area_for_buffer()
    

     # Check if area_gdf is a GeoPandas DataFrame
    if isinstance(area_gdf, gpd.GeoDataFrame):
        selected_geom = area_gdf.iloc[0].geometry
        
    # If area_gdf is a dictionary representing a geometry
    elif isinstance(area_gdf, dict) and 'type' in area_gdf and 'coordinates' in area_gdf:
        # Convert dictionary to a GeoPandas DataFrame
        geom = shape(area_gdf)
        area_gdf = gpd.GeoDataFrame({'geometry': [geom]})
        selected_geom = area_gdf.iloc[0].geometry
    else:
        raise ValueError("Invalid input: area_gdf must be a GeoPandas DataFrame or a valid GeoJSON-like dictionary.")
        
    
    if area_gdf is not None and not area_gdf.empty:
        # Use the geometry of the first (and only) feature in the GeoDataFrame
        selected_geom = area_gdf.iloc[0].geometry
        
        # Reproject the geometry to EPSG:3857 for buffering in meters
        proj = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True).transform
        reprojected_geom = transform(proj, selected_geom)
        
        # Create buffer around the reprojected geometry
        buffer_geom = reprojected_geom.buffer(buffer_distance)
        
        # Reproject the buffer back to EPSG:4326
        proj_back = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326', always_xy=True).transform
        buffer_geom = transform(proj_back, buffer_geom)
        
        # Convert the buffer to a GeoDataFrame
        AREA_BufferA = gpd.GeoDataFrame({'geometry': [buffer_geom]}, crs='epsg:4326')
        
        # Create GeoData layer for the selected area
        selected_geo_data = GeoData(
            geo_dataframe=area_gdf,
            style={
                "color": "black",
                "fillColor": "#3366cc",
                "opacity": 0.05,
                "weight": 1.9,
                "dashArray": "2",
                "fillOpacity": 0.6,
            },
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            name="Selected Area",
        )
        
        # Create GeoData layer for the buffer area
        buffer_geo_data = GeoData(
            geo_dataframe=AREA_BufferA,
            style={
                "color": "black",
                "fillColor": "#ffcc00",
                "opacity": 0.5,
                "weight": 1.9,
                "dashArray": "2",
                "fillOpacity": 0.3,
            },
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            name="Buffer Area",
        )
        
        # Calculate the center of the buffer area
        bounds = AREA_BufferA.total_bounds  # returns (minx, miny, maxx, maxy)
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create a map centered on the buffer area
        buffer_map = Map(center=center, zoom=10, basemap=basemaps.Esri.WorldImagery, layout=Layout(height='600px'))
        
        # Add GeoData layers to the map
        buffer_map.add_layer(selected_geo_data)
        buffer_map.add_layer(buffer_geo_data)
        
        # Fit map to bounds
        sw = [bounds[1], bounds[0]]  # southwest corner (miny, minx)
        ne = [bounds[3], bounds[2]]  # northeast corner (maxy, maxx)
        buffer_map.fit_bounds([sw, ne])
        
        # Add controls to the map
        buffer_map.add_control(LayersControl(position='topright'))
        buffer_map.add_control(FullScreenControl())
        
        # Display the map
        display(buffer_map)
        
        
        def confirm_buffer_selection(button):
            #fetch the area selection type
            global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)
            
            # if its an area that was clicked and selected on the MAP
            if global_area_selection_type and global_area_selection_type == "Select":
                
                # Get the original geometry
                original_geom = area_gdf.iloc[0].geometry
                # Get the buffered geometry
                buffered_geom = AREA_BufferA.iloc[0].geometry
                # Combine original and buffered geometries
                combined_geom = original_geom.union(buffered_geom) 
                # Create a new GeoDataFrame with the combined geometry
                combined_gdf = gpd.GeoDataFrame({"geometry": [combined_geom]}, crs="epsg:4326")
                
                # Convert to GeoJSON-like dictionary
                buffered_area_togeojson = combined_gdf.iloc[0].geometry.__geo_interface__
                # Set the new buffered area as the global selected area
                set_global_result("final_area_selection", combined_gdf, RESULTS)
            else:
                geometry = AREA_BufferA.iloc[0]['geometry']
                # Convert to GeoJSON-like dictionary
                buffered_area_togeojson = geometry.__geo_interface__
                set_global_result("global_selected_area", buffered_area_togeojson, RESULTS)
                # For drawn areas set the new buffered area as the global selected area
                set_global_result("final_area_selection", buffered_area_togeojson, RESULTS)
            
            
                
                
        # Create a button for confirming the buffer as the selection
        confirm_selection_button = Button(description="CONFIRM BUFFER")
        confirm_selection_button.on_click(confirm_buffer_selection)

        # Display the instructions, button, and map
        instructions = HTML("<b>If you would like to proceed with the buffer + site, click  <span style='color:orange'> CONFIRM BUFFER </span> button below. <br> If you would like to change the buffer distance return to section 1.2.1. <p> To exclude site and retain buffer area only, move to section 1.2.3. </b>")
        display(VBox([instructions, confirm_selection_button]))

    else:
        display(HTML("No area selected."))


# Function to confirm buffer distance selection
def on_confirm_button_clicked(b):
    selected_buffer = get_global_result("buffer_distance", RESULTS)
    confirmed_buffer_distance = selected_buffer
    set_global_result("confirmed_buffer_distance", confirmed_buffer_distance, RESULTS)
    

    
style = {'description_width': 'initial'}          
# Text box for custom buffer distance
custom_buffer_distance = BoundedFloatText(
    value=1,
    min=0.001,
    max=100,
    step=0.1,
    description='Custom (km): Maximum = 100km',
    # layout=Layout(width='200px')  # Set the width of the widget
       layout=Layout(width='40%'),
        style=style
)
custom_buffer_distance.layout.display = 'none'  # Hide initially   
        
# Button to confirm buffer distance selection
confirm_button = Button(description="Confirm Buffer Distance")
confirm_button.on_click(on_confirm_button_clicked) 


        
def include_buffer():
    """When called will show options to add buffer to selected site including the site selection."""
    global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)

    # Display widgets
    display(VBox([buffer_distance_options, custom_buffer_distance, confirm_button]))
    # Observe changes
    buffer_distance_options.observe(on_buffer_distance_change, names='value')
    custom_buffer_distance.observe(on_custom_buffer_distance_change, names='value')
    # Initialize buffer_distance variable
    buffer_distance = 100
    confirmed_buffer_distance = buffer_distance  # Initialize confirmed buffer distance
    
    
def buffer_include_selection():
    """Adds set buffer to stored area selection including"""
    # Fetch set selected area 
    polygon_select = polygon_selected()
    # Fetch set selected buffer
    selected_buffer = get_global_result("buffer_distance", RESULTS)
    selected_global_polygon_type = get_global_result("global_selected_polygon_type", RESULTS)
    global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)
    
    # Check if polygon_select is not None and selected_buffer is set
    if polygon_select is not None and selected_buffer:
        # If polygon_select is a GeoPandas DataFrame
        if isinstance(polygon_select, gpd.GeoDataFrame):
            if not polygon_select.empty:
                # NOTE: ensure that only one entry of geopandas dataframe is passed (i.e one selected area)    
                if len(polygon_select) > 1:
                    print("Buffer can only be applied on a single selected area.")
                    return None
                # Convert the first geometry to GeoJSON-like dict
                polygon_select = mapping(polygon_select.iloc[0].geometry)
            else:
                print("The GeoDataFrame is empty. No valid area selected.")
                return None
        # Pass the (possibly converted) polygon_select to create_and_display_buffer_include_selection
        try:
            create_and_display_buffer_include_selection(polygon_select, selected_buffer)
        except Exception as e:
            print("An Error occured applying buffer:", e)
            return None                               
    else:
        print("No area or buffer selection set")
    return None


    
    
# Function to create AREA_BufferB by removing AREA_selection from AREA_BufferA
def create_and_display_buffer_exclude_selection(area_gdf, buffer_distance):
    
    # prepare the area for buffer application
    area_gdf = prepare_area_for_buffer()
    
    
    global AREA_BufferB  # Declare AREA_BufferB as a global variable to store the new buffer area
     # Check if area_gdf is a GeoPandas DataFrame
    if isinstance(area_gdf, gpd.GeoDataFrame):
        selected_geom = area_gdf.iloc[0].geometry
        
    # If area_gdf is a dictionary representing a geometry
    elif isinstance(area_gdf, dict) and 'type' in area_gdf and 'coordinates' in area_gdf:
        # Convert dictionary to a GeoPandas DataFrame
        geom = shape(area_gdf)
        area_gdf = gpd.GeoDataFrame({'geometry': [geom]})
        selected_geom = area_gdf.iloc[0].geometry
    else:
        raise ValueError("Invalid input: area_gdf must be a GeoPandas DataFrame or a valid GeoJSON-like dictionary.")
        
    
    if area_gdf is not None and not area_gdf.empty:
        # Use the geometry of the first (and only) feature in the GeoDataFrame
        selected_geom = area_gdf.iloc[0].geometry
        
        # Reproject the geometry to EPSG:3857 for buffering in meters
        proj = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True).transform
        reprojected_geom = transform(proj, selected_geom)
        
        # Create buffer around the reprojected geometry
        buffer_geom = reprojected_geom.buffer(buffer_distance)
        
        # Reproject the buffer back to EPSG:4326
        proj_back = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326', always_xy=True).transform
        buffer_geom = transform(proj_back, buffer_geom)
        
        # Convert the buffer to a GeoDataFrame
        AREA_BufferA = gpd.GeoDataFrame({'geometry': [buffer_geom]}, crs='epsg:4326')
        bufferA_geom = AREA_BufferA.iloc[0].geometry
    
        # Perform the difference operation
        bufferB_geom = bufferA_geom.difference(selected_geom)

        # Convert the result to a GeoDataFrame
        AREA_BufferB = gpd.GeoDataFrame({'geometry': [bufferB_geom]}, crs='epsg:4326')
    
        # Create GeoData layer for AREA_BufferB
        bufferB_geo_data = GeoData(
            geo_dataframe=AREA_BufferB,
            style={
                "color": "black",
                "fillColor": "#00cc66",
                "opacity": 0.5,
                "weight": 1.9,
                "dashArray": "2",
                "fillOpacity": 0.3,
            },
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            name="Buffer B Area",
        )
    
        # Calculate the center of the buffer area
        bounds = AREA_BufferB.total_bounds  # returns (minx, miny, maxx, maxy)
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
        # Create a map centered on the buffer area
        bufferB_map = Map(center=center, zoom=10, basemap=basemaps.Esri.WorldImagery, layout=Layout(height='600px'))
    
        # Add GeoData layer to the map
        bufferB_map.add_layer(bufferB_geo_data)
    
        # Fit map to bounds
        sw = [bounds[1], bounds[0]]  # southwest corner (miny, minx)
        ne = [bounds[3], bounds[2]]  # northeast corner (maxy, maxx)
        bufferB_map.fit_bounds([sw, ne])
    
        # Add controls to the map
        bufferB_map.add_control(LayersControl(position='topright'))
        bufferB_map.add_control(FullScreenControl())
    
        # Display the map
        display(bufferB_map)

        # # Function to confirm and  buffer selection
        # def confirm_buffer_selection(button):
        #     # geometry = AREA_BufferB.loc[0, 'geometry']
        #     geometry = AREA_BufferB.iloc[0]['geometry']
        #     # Convert to GeoJSON-like dictionary
        #     buffered_area_togeojson = geometry.__geo_interface__
        #     set_global_result("global_selected_area", buffered_area_togeojson, RESULTS)
        
        def confirm_buffer_selection(button):
            #fetch the area selection type
            global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)
            
            # if it's an area that was clicked and selected on the MAP
            if global_area_selection_type and global_area_selection_type == "Select":
                # Get the buffered geometry (which already excludes the original geometry)
                buffered_geom = AREA_BufferB.iloc[0].geometry
                # Create a new GeoDataFrame with the buffered geometry
                buffered_gdf = gpd.GeoDataFrame({"geometry": [buffered_geom]}, crs="epsg:4326")
                
                # Set the new buffered area as the global selected area
                set_global_result("final_area_selection", buffered_gdf, RESULTS)
            else:
                # For drawn areas, convert to GeoJSON-like dictionary
                geometry = AREA_BufferB.iloc[0]['geometry']
                buffered_area_togeojson = geometry.__geo_interface__
                set_global_result("global_selected_area", buffered_area_togeojson, RESULTS)
                # For drawn areas set the new buffered area as the global selected area
                set_global_result("final_area_selection", buffered_area_togeojson, RESULTS)
            
            


        # Create a button for confirming the buffer as the selection
        confirm_selection_button = Button(description="CONFIRM BUFFER")
        confirm_selection_button.on_click(confirm_buffer_selection)

        # Display the instructions, button, and map
        instructions = HTML("<b>If applied buffer excluding site selection is good, click  <span style='color:orange'> CONFIRM BUFFER </span> button below. <br> To make changes to the site selection or buffer return to section - 1.1.3 Select site on a map. </b>")
        display(VBox([instructions, confirm_selection_button]))




def buffer_exclude_selection():
    """Adds set buffer to stored area selection excluding the selection"""
    # Fetch set selected area 
    polygon_select = polygon_selected()
    # Fetch set selected buffer
    selected_buffer = get_global_result("buffer_distance", RESULTS)
    selected_global_polygon_type = get_global_result("global_selected_polygon_type", RESULTS)
    global_area_selection_type = get_global_result("global_area_selection_type", RESULTS)
    
    # Check if polygon_select is not None and selected_buffer is set
    if polygon_select is not None and selected_buffer:
        # If polygon_select is a GeoPandas DataFrame
        if isinstance(polygon_select, gpd.GeoDataFrame):
            if not polygon_select.empty:
                # NOTE: ensure that only one entry of geopandas dataframe is passed (i.e one selected area)    
                if len(polygon_select) > 1:
                    print("Buffer can only be applied on a single selected area.")
                    return None

                # Convert the first geometry to GeoJSON-like dict
                polygon_select = mapping(polygon_select.iloc[0].geometry)
            else:
                print("The GeoDataFrame is empty. No valid area selected.")
                return None
            
        # Pass the (possibly converted) polygon_select to create_and_display_buffer_include_selection
        try:
             create_and_display_buffer_exclude_selection(polygon_select, selected_buffer)
        except Exception as e:
            print("An Error occured applying buffer:", e)
            return None  
                                    
    else:
        print("No area or buffer selection set")
        return None


# ================================= Save selection as a shapefile ==========================================

def create_shape_file(selected_polygon, name, destination_path):
    """
    Save a selected polygon as an ESRI Shapefile (with .cpg and .qmd sidecar files).

    Parameters
    ----------
    selected_polygon : geopandas.GeoDataFrame | dict (GeoJSON) | shapely geometry
        The area to save. Must be a Polygon or MultiPolygon (Points are rejected).
    name : str
        Base name used for the output files (e.g. "NorthSea_Boundary").
    destination_path : str
        Existing directory in which to write the shapefile.

    Raises
    ------
    ValueError
        If no polygon is defined, if the geometry is a Point/non-polygon type,
        or if the destination directory does not exist.

    Returns
    -------
    list[str]
        Sorted list of the file names that were created.
    """
    # 1. selected_polygon must be defined
    if selected_polygon is None:
        raise ValueError(
            "selected_polygon is not defined. Select/draw and confirm an area before saving."
        )

    # 2. Normalise the input into a GeoDataFrame (EPSG:4326)
    if isinstance(selected_polygon, gpd.GeoDataFrame):
        if selected_polygon.empty:
            raise ValueError("selected_polygon is an empty GeoDataFrame. Nothing to save.")
        gdf = selected_polygon.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
    elif isinstance(selected_polygon, dict) and "type" in selected_polygon and "coordinates" in selected_polygon:
        geom = shape(selected_polygon)
        gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[geom], crs="EPSG:4326")
    elif hasattr(selected_polygon, "geom_type"):  # a shapely geometry
        gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[selected_polygon], crs="EPSG:4326")
    else:
        raise ValueError(
            "selected_polygon must be a GeoDataFrame, a GeoJSON-like dict, or a shapely geometry."
        )

    # 3. Geometry type must be a Polygon or MultiPolygon (no Points / lines)
    allowed_types = {"Polygon", "MultiPolygon"}
    geom_types = set(gdf.geometry.geom_type.unique())
    if "Point" in geom_types or "MultiPoint" in geom_types:
        raise ValueError("Selected geometry is a Point. Selection must be a Polygon or MultiPolygon.")
    if not geom_types.issubset(allowed_types):
        raise ValueError(
            f"Selected geometry type(s) {sorted(geom_types)} not supported. "
            "Only Polygon and MultiPolygon are accepted."
        )

    # 4. Destination directory must exist
    dest_dir = Path(destination_path)
    if not dest_dir.is_dir():
        raise ValueError(f"Destination path does not exist: {destination_path}")

    # 5. Write the shapefile (creates .shp, .shx, .dbf, .prj)
    shapefile_path = dest_dir / f"{name}.shp"
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="UTF-8")

    # ensure .cpg with encoding
    (dest_dir / f"{name}.cpg").write_text("UTF-8")

    # ensure .qmd metadata file
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    (dest_dir / f"{name}.qmd").write_text(
        f"Name: {name}\n"
        f"BBox: {bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}\n"
        f"CRS: EPSG:4326"
    )

    # 6. Confirm and print created files
    created = sorted(p.name for p in dest_dir.glob(f"{name}.*"))
    print(created)
    return created













