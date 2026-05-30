import os
import pickle
import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

COOKIE_FILE = (
    "linkedin_cookies.pkl"  # Use get_cookies to retrieve pickle file the first time
)


def load_cookies(driver, cookie_file):
    """
    Function: load_cookies retrieves cookies saved in the pickle file

    Args:
        driver: Selenium driver that was initialised
        cookie_file: OS path to cookie pickle file

    Description: Retrieves cookies and loads them in the driver
    """
    if not os.path.exists(cookie_file):
        raise FileNotFoundError(
            f"Cookie file '{cookie_file}' not found. Please login manually and save cookies first by running get_cookies.py"
        )

    with open(cookie_file, "rb") as f:
        cookies = pickle.load(f)
    for cookie in cookies:
        # Remove 'sameSite' if present — sometimes causes issues
        cookie.pop("sameSite", None)
        driver.add_cookie(cookie)


def initialise_driver(cookie_file):
    """
    Function: initialises the driver to be used

    Args:
        cookie_file: OS path to cookie pickle file

    Description: Initialises the driver, launches LinkedIn and loads the cookies
    """
    chrome_options = Options()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--high-dpi-support=1")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-font-subpixel-positioning")
    chrome_options.add_argument("--disable-lcd-text")
    if os.environ.get("SESSIONNAME") != "Console":
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
    else:
        chrome_options.add_argument("--start-maximized")

    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )

    # Suppress console noise
    chrome_options.add_argument("--log-level=3")  # ERROR level only
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-logging"]
    )  # Hide DevTools logs
    chrome_options.add_experimental_option(
        "prefs",
        {
            "profile.default_content_setting_values.notifications": 2,  # Disable browser notifications
            "profile.default_content_setting_values.media_stream_mic": 2,
            "profile.default_content_setting_values.media_stream_camera": 2,
            "profile.default_content_setting_values.geolocation": 2,
        },
    )

    driver.get(
        "https://www.linkedin.com"
    )  # Open normal linkedin first to confirm/set cookies
    time.sleep(3)

    load_cookies(driver, cookie_file)
    print("Cookies loaded")

    return driver


def scraper(driver):
    """
    Function: scraper will scrape the metadata for the board

    Args:
        driver: Selenium driver that was initialised

    Description: Scrapes the metadata of the queens board for use

    Returns: dict: {'board_size': int, 'board': List[List[int]]}
    """

    driver.get("https://www.linkedin.com/games/queens")
    driver.execute_script("document.body.style.zoom='100%'")
    time.sleep(1)

    board = driver.find_element(By.ID, "queens-game-board")

    style = board.get_attribute("style")
    row_match = re.search(r"--rows:\s*(\d+)", style)
    col_match = re.search(r"--cols:\s*(\d+)", style)
    if not row_match or not col_match:
        raise ValueError(f"Invalid style string: {style}")
    rows = int(row_match.group(1))
    cols = int(col_match.group(1))
    assert rows == cols, "Grid should be square"

    size = rows
    colour_region_matrix = [[None for _ in range(size)] for _ in range(size)]

    """
    # OLD CODE. DELETED DUE TO 5s SLOWER AS IT WAS MULTIPLE QUERIES IN A LOOP.
    cells = board.find_elements(By.CLASS_NAME, "queens-cell-with-border")

    for cell in cells:
        class_attr = cell.get_attribute("class")
        aria = cell.get_attribute("aria-label")

        color_match = re.search(r"cell-color-(\d+)", class_attr)
        color_index = int(color_match.group(1)) if color_match else None

        aria_match = re.search(r"row (\d+), column (\d+)", aria)
        if not aria_match:
            continue

        row = int(aria_match.group(1)) - 1  # 1-indexed to 0-indexed
        col = int(aria_match.group(2)) - 1

        colour_region_matrix[row][col] = color_index
    """

    # Used LLM to help Create similar query but in JavaScript for speed
    script = """
        return Array.from(document.querySelectorAll(".queens-cell-with-border")).map(cell => {
            return {
                classAttr: cell.className,
                aria: cell.getAttribute("aria-label")
            };
        });
    """
    cell_data = driver.execute_script(script)

    for entry in cell_data:
        class_attr = entry["classAttr"]
        aria = entry["aria"]

        colour_match = re.search(r"cell-color-(\d+)", class_attr)
        colour_index = int(colour_match.group(1)) if colour_match else None

        aria_match = re.search(r"row (\d+), column (\d+)", aria)
        if not aria_match:
            continue

        row = int(aria_match.group(1)) - 1
        col = int(aria_match.group(2)) - 1

        colour_region_matrix[row][col] = colour_index

    # We now quit in the inputter
    # driver.quit()

    return {"board_size": size, "board": colour_region_matrix}
