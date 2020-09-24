
import torch
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from glob import glob

def parallel_coordinates(model_save_dir, metric="loss"):
    # DO NOT CHANGE ANYTHING IN THIS FUNCTION
    # This function take a folder path and loads all model files,
    # extracts hyperparamaters and the given score, and creates a 
    # parallell coordination plot to display the scores for each hyperparamater
    
    model_files = glob(model_save_dir+ "/*.pt")
    
    rows = []
    for model_file in model_files:
        model_dict = checkpoint = torch.load(model_file)
        hyperparamaters = model_dict["hyperparamaters"]
        score_dict["loss"] = model_dict["loss"]
        score_dict.update(model_dict["hyperparamaters"])

        row = hyperparamaters
        row[metric] = score_dict[metric]
        rows.append(row)

    df = pd.DataFrame(rows)

    y_dims = []
    for cname,cdata in df.iteritems():
        values = list(cdata.values)

        add_text_ticks = False
        if isinstance(values[0], str):
            ticktexts = sorted(set(values))
            ticktext2id = {v:i for i,v  in enumerate(ticktexts)}
            tickvals = list(ticktext2id.values())
            values = [ticktext2id[v] for v in values]
            add_text_ticks = True
        
        max_v = np.max(values)
        min_v = np.min(values)

        y_dict = {
                    "values":values,
                    "label":cname,
                    "range": [min_v, max_v]
                    }
        
        if add_text_ticks:
            y_dict["ticktext"] = ticktexts
            y_dict["tickvals"] = tickvals
        
        y_dims.append(y_dict)


    fig = go.Figure(data=go.Parcoords(
                                        line = dict(
                                                    color = df[metric],
                                                    colorscale = 'Electric',
                                                    showscale = True,
                                                    cmin = df[metric].min(),
                                                    cmax = df[metric].max()
                                                    ),
                                        dimensions = y_dims,
                                    )
                     )     
    fig.show()