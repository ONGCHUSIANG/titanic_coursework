from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
import pandas as pd

class TitanicVisualizer:
    def __init__(self):
        pass

    def save_survival_chart(self, data):
        # 1. Group data by Passenger Class (Pclass) to find relationships
        classes = ['1st Class', '2nd Class', '3rd Class']
        outcomes = ['Died', 'Survived']
        
        # Calculate the exact numbers dynamically from your dataframe
        died_counts = [
            len(data[(data['Pclass'] == 1) & (data['Survived'] == 0)]),
            len(data[(data['Pclass'] == 2) & (data['Survived'] == 0)]),
            len(data[(data['Pclass'] == 3) & (data['Survived'] == 0)])
        ]
        
        survived_counts = [
            len(data[(data['Pclass'] == 1) & (data['Survived'] == 1)]),
            len(data[(data['Pclass'] == 2) & (data['Survived'] == 1)]),
            len(data[(data['Pclass'] == 3) & (data['Survived'] == 1)])
        ]

        # 2. Package data for Bokeh
        data_dict = {
            'classes': classes,
            'Died': died_counts,
            'Survived': survived_counts
        }
        source = ColumnDataSource(data=data_dict)
        
        output_file("survival_chart_interactive.html", title="Titanic Survival Analysis")

        # 3. Create the Canvas
        p = figure(x_range=classes, height=500, width=750, 
                   title="Titanic Survival Breakdown by Passenger Class",
                   toolbar_location="right", tools="pan,wheel_zoom,box_zoom,reset,save")

        # 4. Draw the Stacked Bar Chart
        colors = ['#7f8c8d', '#2980b9'] # Slate Grey for Died, Steel Blue for Survived
        
        renderers = p.vbar_stack(outcomes, x='classes', width=0.55, color=colors, 
                                 source=source, legend_label=outcomes)

        # 5. Add Advanced Hover Tooltips
        # This will show the exact count based on where the user places their mouse
        hover = HoverTool(tooltips=[
            ("Passenger Class", "@classes"),
            ("Status", "$name"),
            ("Count", "@$name passengers")
        ])
        p.add_tools(hover)

        # 6. Professional Polish & Formatting
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xgrid.grid_line_color = None
        p.axis.minor_tick_line_color = None
        p.outline_line_color = None
        
        # Format the legend
        p.legend.location = "top_left"
        p.legend.orientation = "horizontal"
        p.legend.background_fill_alpha = 0.5
        
        # Format the fonts
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label = "Passenger Ticket Class"
        p.yaxis.axis_label = "Total Number of Passengers"
        p.xaxis.axis_label_text_font_style = "bold"
        p.yaxis.axis_label_text_font_style = "bold"

        # 7. Show the upgraded chart
        show(p) 
        print("Advanced interactive survival chart saved as 'survival_chart_interactive.html'")
    def save_feature_importance_chart(self, model, feature_names):
        """Extracts and plots the brain of the Random Forest."""
        import pandas as pd
        
        # 1. Extract importances from the model
        importances = model.feature_importances_
        
        # 2. Create a DataFrame and sort them from most to least important
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        df = df.sort_values(by='Importance', ascending=True) # Ascending so biggest is at the top of the chart
        
        # Keep only the top 10 features so the chart isn't cluttered
        df = df.tail(10)
        
        source = ColumnDataSource(df)
        output_file("feature_importance.html", title="Model Feature Importance")

        # 3. Create a Horizontal Bar Chart
        p = figure(y_range=df['Feature'], height=500, width=700, 
                   title="What mattered most for survival? (Top 10 Features)",
                   toolbar_location="right", tools="pan,wheel_zoom,box_zoom,reset,save")

        p.hbar(y='Feature', right='Importance', height=0.6, source=source, 
               color="#2ecc71", line_color="white", hover_fill_alpha=0.8)

        # 4. Add Tooltips
        hover = HoverTool(tooltips=[
            ("Feature", "@Feature"),
            ("Importance Score", "@Importance{0.000}") # Format to 3 decimal places
        ])
        p.add_tools(hover)

        # 5. Formatting
        p.x_range.start = 0
        p.ygrid.grid_line_color = None
        p.xaxis.axis_label = "Importance Score (Higher = More Important)"
        p.yaxis.axis_label_text_font_style = "bold"
        p.outline_line_color = None

        show(p)
        print("Feature importance chart saved as 'feature_importance.html'")