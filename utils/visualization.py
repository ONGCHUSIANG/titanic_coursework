from bokeh.plotting import figure, show, save, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap

class TitanicVisualizer:
    def __init__(self):
        pass

    # 1. Renamed to match main.py and added 'data' parameter
    def save_survival_chart(self, data):
        
        # 2. Dynamically count survivors from the passed dataframe
        # (Assuming your dataframe has a 'Survived' column with 0s and 1s)
        died_count = len(data[data['Survived'] == 0])
        survived_count = len(data[data['Survived'] == 1])
        
        categories = ['Died', 'Survived']
        counts = [died_count, survived_count] 
        
        source = ColumnDataSource(data=dict(Status=categories, Counts=counts))
        output_file("survival_chart_interactive.html", title="Titanic Survival Rates")
        color_map = factor_cmap('Status', palette=['#7f8c8d', '#2980b9'], factors=categories)

        p = figure(x_range=categories, height=450, width=600, 
                   title="Titanic Passenger Survival Counts",
                   toolbar_location="right", tools="pan,wheel_zoom,box_zoom,reset,save")

        p.vbar(x='Status', top='Counts', width=0.5, source=source, 
               line_color="white", fill_color=color_map, 
               hover_fill_alpha=0.8, hover_line_color="#333333")

        hover = HoverTool()
        hover.tooltips = [("Status", "@Status"), ("Passengers", "@Counts")]
        p.add_tools(hover)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        p.yaxis.axis_label = "Number of Passengers"
        p.xaxis.axis_label = "Passenger Status"
        p.outline_line_color = None

        # 3. Saves the HTML file and opens it in your default web browser
        show(p) 
        print("Interactive survival chart saved as 'survival_chart_interactive.html'")