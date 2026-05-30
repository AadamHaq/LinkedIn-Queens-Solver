import re
import time

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

# Only need to input as it is assumed we are still on the page from the scraper


def input_solution(driver, solution):
    """
    Function: input_solution places the queens in the grid

    Args:
        driver: Selenium driver that was initialised
        solution: List[Tuple(int, int)] rows and columns of the queens in the solution. SOLUTION MUST BE 1-INDEXED DUE TO LINKEDIN

    Description: Completes the board!

    Returns: None
    """

    board = driver.find_element(By.ID, "queens-game-board")
    cells = board.find_elements(By.CSS_SELECTOR, '[data-testid^="cell-"]')
    actions = ActionChains(driver)

    for row, col in solution:
        found = False
        for cell in cells:
            aria = cell.get_attribute("aria-label")
            if not aria:
                continue

            match = re.search(r"row (\d+), column (\d+)", aria)
            if match:
                r = int(match.group(1))
                c = int(match.group(2))
                if r == row and c == col:
                    print(f"Placing queen at row {r}, col {c}")
                    actions.move_to_element(cell).pause(0.05).double_click().perform()
                    time.sleep(0.05)
                    found = True
                    break
        if not found:
            print(f"Could not find cell for ({row}, {col})")
