import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
filename = 'routeTRUCKS5ITERATIONS1000AL1BE2Q1.3RHO0.15_W[0.4,0.4,0.2,0.1]_20241116_152056.csv'
df = pd.read_csv(f"data/routes/{filename}")

# Create a single figure with multiple subplots
fig, axs = plt.subplots(3, 2, figsize=(8, 8))  # 3 rows, 2 columns of subplots
fig.suptitle('Analysis of Route Efficiency Metrics', fontsize=16)

# 1. Graph MaxDistance vs Iteration
axs[0, 0].plot(df['Iteration'], df['TotalDistance'], label='Total Distance', color='blue', marker='')
axs[0, 0].set_title('Max Distance per Iteration')
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('Max Distance (m)')
axs[0, 0].grid(True)
axs[0, 0].legend()

# 2. Graph MaxTime vs Iteration
axs[0, 1].plot(df['Iteration'], df['MaxTime'], label='Max Time', color='green', marker='')
axs[0, 1].set_title('Max Time per Iteration')
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Max Time (s)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# 3. Graph MaxWasteCollected vs Iteration
axs[1, 0].plot(df['Iteration'], df['TotalWasteCollected'], label='Total Waste Collected', color='red', marker='')
axs[1, 0].set_title('Max Waste Collected per Iteration')
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Max Waste Collected (kg)')
axs[1, 0].grid(True)
axs[1, 0].legend()

# 4. Graph AvgScore vs Iteration
axs[1, 1].plot(df['Iteration'], df['AvgScore'], label='Avg Score', color='purple', marker='')
axs[1, 1].set_title('Avg Score per Iteration')
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Avg Score')
axs[1, 1].grid(True)
axs[1, 1].legend()

# 5. Graph Time vs Iteration
axs[2, 0].plot(df['Iteration'], df['Time'], label='Time', color='orange', marker='')
axs[2, 0].set_title('Time per Iteration')
axs[2, 0].set_xlabel('Iteration')
axs[2, 0].set_ylabel('Time (s)')
axs[2, 0].grid(True)
axs[2, 0].legend()

# 6. Number of Nodes Visited vs Iteration
axs[2, 1].plot(df['Iteration'], df['SumofNodesVisited'], label='Nodes Visited', color='black', marker='')
axs[2, 1].set_title('Nodes Visited per Iteration')
axs[2, 1].set_xlabel('Iteration')
axs[2, 1].set_ylabel('Nodes')
axs[2, 1].grid(True)
axs[2, 1].legend()

# Hide the last subplot if not used (3x2 layout has an extra slot)


# Adjust layout to avoid overlapping
plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Adjust space for the main title

# Show the plot
plt.show()
