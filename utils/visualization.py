# utils/visualization.py
import matplotlib.pyplot as plt

class TitanicVisualizer:
    def save_survival_chart(self, df):
        """Creates a bar chart of survivors and saves it."""
        survival_counts = df['Survived'].value_counts()
        
        # Plotting the data
        survival_counts.plot(kind='bar', color=['red', 'green'])
        plt.title('Titanic Survival Counts (0 = Died, 1 = Survived)')
        plt.xlabel('Status')
        plt.ylabel('Number of Passengers')
        
        # Save the visualization as a file
        plt.savefig('survival_chart.png')
        print("✅ Visualization saved successfully as 'survival_chart.png'!")