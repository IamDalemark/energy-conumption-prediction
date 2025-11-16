import numpy as np
import pandas as pd
def preprocessDataflow(df):
    mapBuildingTypes = {
        "Residential": 0 , "Commercial": 0.5, "Industrial": 1
    }
    df["BuildingType"] = df["BuildingType"].map(mapBuildingTypes)
    return df