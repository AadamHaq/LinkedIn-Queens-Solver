import argparse
import time

from Logic.computer_vision import computer_vision, get_image
from Logic.game_inputter import input_solution
from Logic.game_scraper import initialise_driver, scraper
from Logic.share_score_after_play import share_score
from Logic.Solvers.naive_backtracking import backtracking


def main(cookie_file, name, solver="backtracking"):
    """
    Function: main runs all other functions for the game

    Args:
        cookie_file: .pkl file that can be retrieved by running get_cookies.py
        name: Name of group chat that the score will be sent to
        solver: "backtracking" (default) or "imitation" (trained policy network)

    Description: Runs all other functions into one seamless solution then quits the driver

    Returns: None
    """

    driver = initialise_driver(cookie_file)

    try:
        path = "queens_board.png"
        get_image(driver, path)
        board = computer_vision(path)
        N = len(board)
        print(board)

    except Exception:
        data = scraper(driver)

        board = data["board"]
        N = data["board_size"]
        print(board)
    if solver == "imitation":
        # Imported lazily so the default path never pays the torch import cost
        from Logic.Solvers.imitation_solver import imitation

        solution = imitation(board, N)
    else:
        solution = backtracking(board, N)

    print(solution)
    solution_1_indexed = [(r + 1, c + 1) for r, c in solution]
    input_solution(driver, solution_1_indexed)

    time.sleep(5)
    share_score(driver, name)

    driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinkedIn Queens solver")
    parser.add_argument(
        "--solver",
        choices=["backtracking", "imitation"],
        default="backtracking",
        help="which solver works out the queen placements",
    )
    args = parser.parse_args()

    COOKIE_FILE = "linkedin_cookies.pkl"
    name = "Queens + Zip Daily"
    main(COOKIE_FILE, name, solver=args.solver)
