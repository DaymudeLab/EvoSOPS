from PIL import Image
import numpy as np

grids = []

with open("./output/Brid_GM_1_sizes_1_trials_gran_10.log", "r") as outputFile:
    readingGrid = False
    currentGrid = []
    for line in outputFile:
        if "SOPS grid" in line:
            readingGrid = True
        elif "Fitness:" in line or "Edge Count:" in line:
            if len(currentGrid) > 0:
                grids.append(currentGrid)
                currentGrid = []
            readingGrid = False
        elif readingGrid:
            currentLine = []
            for character in line.strip().split("  "):
                currentLine.append(int(character))
            currentGrid.append(currentLine)

color_palette = [(255, 255, 255),  # White for 0
                  (255, 0, 0),  # Red for 1
                  (0, 0, 255),  # Blue for 2
                  (128, 0, 128), # Purple for 3
                  (255, 255, 0), # Yellow for 4
                  (0, 0, 0)]  # Black for 5

def create_gif_from_grids(grids, filename="./output/output.gif", fps=5, scale=4, size=(1024, 1024)):
    frames = []
    finalFrameRepeatAmount = 10


    for gridIndex in range(len(grids) + finalFrameRepeatAmount - 1):
        if gridIndex >= len(grids):
            gridIndex = len(grids) - 5

        grid = grids[gridIndex]
        grid_array = np.array(grid)

        # Scale up the grid using pixel repetition
        scaled_array = np.repeat(np.repeat(grid_array, scale, axis=0), scale, axis=1)

        image = Image.fromarray(scaled_array.astype(np.uint8), 'P')
        image.putpalette(np.array(color_palette, dtype='uint8').flatten())
        frames.append(image.resize(size, resample=Image.NEAREST))

    frames[0].save(
        filename, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True, 
        duration=1000 // fps,
        loop=0  
    )

# Example usage with a larger size
create_gif_from_grids(grids, fps=5, scale=10) 