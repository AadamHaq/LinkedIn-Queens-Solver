"""
Sources: https://www.geeksforgeeks.org/opencv-python-tutorial/
https://towardsdatascience.com/solving-linkedin-queens-game-cfeea7a26e86/ - Used as a starting inspiration but quickly diverged from this solution


Old version

import cv2 as cv

# Load the image
image = cv.imread("Queens/example_board.png")

# Resize to standardise (may remove)
resized = cv.resize(image, (350, 350))

# Convert to grayscale
gray = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
# Create threshold to detect cell boundaries
_, thresh = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)

# Find contours which represent the blocks
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a mask for each unique color
unique_colors = {}
board_size = 7  # 7x7 grid in example, need to find a generalisation
cell_height = resized.shape[0] // board_size
cell_width = resized.shape[1] // board_size
board = [[-1 for i in range(board_size)] for j in range(board_size)] # Fi.l board with -1

color_index = 0

# Loop through each cell
for row in range(board_size):
    for col in range(board_size):
        # Get the center pixel of the cell
        y = row * cell_height + cell_height // 2 # Bottom of cell + 1/2 the height to get middle
        x = col * cell_width + cell_width // 2
        color = tuple(resized[y, x])

        # Register new color if not already
        if color not in unique_colors:
            unique_colors[color] = color_index
            color_index += 1

        board[row][col] = unique_colors[color]

print(board)

Output:
[[0, 0, 0, 1, 1, 1, 1],
[2, 0, 3, 3, 3, 1, 1],
[2, 0, 0, 0, 3, 1, 4],
[2, 2, 3, 3, 3, 1, 4],
[5, 5, 3, 6, 6, 4, 4],
[5, 4, 3, 3, 3, 4, 4],
[4, 4, 4, 4, 4, 4, 4]]
"""

import os
import time

import cv2 as cv
import numpy as np
from Logic.game_scraper import initialise_driver
from selenium.webdriver.common.by import By


def get_image(driver, path):
    driver.get("https://www.linkedin.com/games/queens")
    driver.execute_script("document.body.style.zoom='100%'")
    time.sleep(1)

    board = driver.find_element(By.ID, "queens-game-board")

    board.screenshot(path)
    print("Screenshot obtained")


def sample_patch(img, x, y, r=4):
    """
    Sample a small square patch around (x, y) and return mean colour.
    """
    h, w, _ = img.shape
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)
    patch = img[y1:y2, x1:x2]
    return np.mean(patch.reshape(-1, 3), axis=0)


def find_colour_id(colour, colour_map, tol=15):
    """
    Find an existing colour within tolerance, otherwise return None.
    """
    for cid, ref in colour_map.items():
        if np.linalg.norm(colour - ref) < tol:
            return cid
    return None


def computer_vision(path):
    # Load image and resize to standardise
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    resized = cv.resize(image, (350, 350))

    # Convert to grayscale and edge detection
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # Uses Canny algorithm

    # Use Hough Transform to detect straight lines in the image
    # Converts edge points from Canny alg. into votes and if a certain line gets many votes, it is a line
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    # Separate vertical and horizontal lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 10:  # Vertical Line
            vertical_lines.append(x1)
        elif abs(y1 - y2) < 10:  # Horizontal Line
            horizontal_lines.append(y1)

    # Cluster close lines as some lines may be repeated
    def cluster_lines(lines, threshold=10):
        lines = sorted(set(lines))  # Removes duplicates and sorts
        clustered = []
        for line in lines:
            if not clustered or abs(line - clustered[-1]) > threshold:
                clustered.append(line)
        return clustered

    cols = cluster_lines(vertical_lines)
    rows = cluster_lines(horizontal_lines)

    num_rows = len(rows) - 1
    num_cols = len(cols) - 1

    if num_rows != num_cols:
        raise ValueError(
            f"Detected board is not square: {num_rows} rows, {num_cols} cols"
        )

    size = num_rows

    # Initialize board with -1
    board = [[-1 for _ in range(num_cols)] for _ in range(num_rows)]

    # Assign colour IDs based on center of each cell
    unique_colours = {}  # colour_id -> mean colour
    colour_index = 0

    for i in range(size):
        for j in range(size):
            y = (rows[i] + rows[i + 1]) // 2
            x = (cols[j] + cols[j + 1]) // 2

            colour = sample_patch(resized, x, y)

            cid = find_colour_id(colour, unique_colours)
            if cid is None:
                cid = colour_index
                unique_colours[cid] = colour
                colour_index += 1

            board[i][j] = cid

    unique_regions = len(set(v for row in board for v in row))
    if unique_regions != size:
        raise ValueError(f"CV failed: expected {size} regions, got {unique_regions}")

    return board


if __name__ == "__main__":
    # Creating training data for RL

    cookie_file = "linkedin_cookies.pkl"
    driver = initialise_driver(cookie_file)
    # get_image(driver)
    # board = computer_vision("board_screenshot.png")
    # print(board)
    img_save = "Queens/Logic/Solvers/RL_Resources/Training_Data/Images"
    os.makedirs(img_save, exist_ok=True)
    txt_save = "Queens/Logic/Solvers/RL_Resources/Training_Data/Metadata"
    os.makedirs(txt_save, exist_ok=True)

    for i in range(1, 387):
        if i in [4, 7, 9, 13, 14, 18, 19, 20]:
            print("Skipped")
            continue
        img_path = os.path.join(img_save, f"queens_board_{i}.png")
        txt_path = os.path.join(txt_save, f"queens{i}.txt")

        driver.get(f"https://queensgame.vercel.app/level/{i}")
        driver.execute_script("document.body.style.zoom='80%'")
        time.sleep(0.5)
        board_elem = driver.find_element(By.CLASS_NAME, "board")
        board_elem.screenshot(img_path)

        board = computer_vision(img_path)
        print(board)

        # Save the board matrix to text file
        with open(txt_path, "w") as f:
            f.write("queens_board = [\n")
            for row in board:
                f.write(f"  {row},\n")
            f.write("]\n")
