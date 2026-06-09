import os
import random
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def time_to_seconds(t):
    """Convert mm:ss string to seconds (or NaN if missing)."""
    if pd.isna(t) or t == "":
        return np.nan
    try:
        mins, secs = t.split(":")
        return int(mins) * 60 + int(secs)
    except Exception:
        return np.nan


def get_fun_fact(df):
    # --- Load aliases from .env ---
    load_dotenv()
    alias_map = {}
    for key, value in os.environ.items():
        if key.startswith("ALIAS_"):
            name_parts = key.replace("ALIAS_", "").split("_")
            first_name = name_parts[0].capitalize()  # only first name
            alias_map[value] = first_name

    players = [p for p in df.columns if p not in ["Day", "DoW", "Backtracking"]]

    # Convert all times to seconds
    df_secs = df.copy()
    for p in players:
        df_secs[p] = df_secs[p].apply(time_to_seconds)

    # Use most recent row as "today"
    today = df_secs.iloc[-1]
    # last14 = df_secs.iloc[-15:-1]  # last 14 days before today

    fun_facts = []

    # --- Fastest solve today ---
    today_valid = today[players].dropna()
    if not today_valid.empty:
        fastest_time = today_valid.min()
        fastest_players = today_valid[today_valid == fastest_time].index.tolist()
        t_fmt = f"{int(fastest_time // 60)}:{int(fastest_time % 60):02d}"
        if len(fastest_players) == 1:
            name = alias_map.get(fastest_players[0], fastest_players[0])
            fun_facts.append(f"🏆 Fastest today: {name} with {t_fmt}")
        else:
            names = [alias_map.get(p, p) for p in fastest_players]
            name_str = ", ".join(names[:-1]) + " and " + names[-1]
            fun_facts.append(f"🏆 Joint fastest today: {name_str} with {t_fmt}")

    # --- Bad day check (> 5:00) ---
    if not today_valid.empty:
        bad_day = today_valid[today_valid > 300]  # 300 seconds = 5 mins
        if not bad_day.empty:
            # Get all names using alias_map
            names = [alias_map.get(player, player) for player in bad_day.index]
            if len(names) == 1:
                name_str = names[0]
            else:
                name_str = ", ".join(names[:-1]) + " and " + names[-1]
            # Use the maximum time among the bad players for display
            max_secs = bad_day.max()
            haris_alias = os.getenv("ALIAS_HARIS_K", "HK")
            haris_bad = haris_alias in bad_day.index
            if haris_bad and len(bad_day) == 1:
                suffix = "... well, that one makes sense 😅"
            elif haris_bad:
                suffix = "... Haris we get, but the rest have no excuse 😅"
            else:
                suffix = "... maybe Haris played on their phone"
            fun_facts.append(
                f"😬 {name_str} had a bad day with a time of up to {int(max_secs // 60)}:{int(max_secs % 60):02d}{suffix}"
            )

    # # --- Most improved (today vs average of last 14 days) ---
    # if not today_valid.empty:
    #     improvements = {}
    #     for p in players:
    #         if not pd.isna(today[p]):
    #             past_avg = last14[p].dropna().mean()
    #             if past_avg and past_avg > 0:
    #                 diff = past_avg - today[p]
    #                 improvements[p] = diff
    #     if improvements:
    #         most_improved = max(improvements, key=improvements.get)
    #         gain = improvements[most_improved]
    #         if gain > 0:
    #             name = alias_map.get(most_improved, most_improved)
    #             fun_facts.append(f"📈 Most improved: {name}, {int(gain)}s faster than his 2-week average!")

    # --- Best recent average (last 14 days) ---
    # averages = {p: last14[p].dropna().mean() for p in players}
    # averages = {p: v for p, v in averages.items() if v and not np.isnan(v)}
    # if averages:
    #     best_avg_player = min(averages, key=averages.get)
    #     best_avg_time = averages[best_avg_player]
    #     name = alias_map.get(best_avg_player, best_avg_player)
    #     fun_facts.append(f"🔥 Best 2-week average: {name} with {int(best_avg_time//60)}:{int(best_avg_time%60):02d}")

    # --- Current win streaks (consecutive days with valid times) ---
    winners = df_secs[players].idxmin(axis=1)

    # Track current consecutive win streaks for each player
    current_streaks = {p: 0 for p in players}
    for p in players:
        streak = 0
        # iterate backwards through days
        for w in reversed(winners):
            if w == p:
                streak += 1
            else:
                break
        current_streaks[p] = streak

    # Find player with the current longest win streak
    best_player = max(current_streaks, key=lambda k: current_streaks[k])
    if current_streaks[best_player] > 1:  # only count streaks > 1
        name = alias_map.get(best_player, best_player)
        fun_facts.append(
            f"🔥 {name} is on a {current_streaks[best_player]}-day win streak!"
        )

    # --- Big drop check (yesterday 1st → today last) ---
    if len(df_secs) >= 2:
        yesterday = df_secs.iloc[-2][players].dropna()
        if not yesterday.empty and not today_valid.empty:
            yesterday_winner = yesterday.idxmin()
            today_loser = today_valid.idxmax()
            if yesterday_winner == today_loser:
                name = alias_map.get(yesterday_winner, yesterday_winner)
                fun_facts.append(
                    f"⬇️ {name} went from 1st to last overnight... someone had a bad sleep!"
                )

    # --- Close race (top two within 1s) ---
    if len(today_valid) >= 2:
        today_sorted = today_valid.sort_values()
        gap = int(today_sorted.iloc[1]) - int(today_sorted.iloc[0])
        if 0 < gap <= 1:
            p1 = alias_map.get(today_sorted.index[0], today_sorted.index[0])
            p2 = alias_map.get(today_sorted.index[1], today_sorted.index[1])
            fun_facts.append(
                f"⚡ {p1} pipped {p2} by just {gap}s today — absolute photo finish! 📸"
            )

    # --- Comeback (yesterday last → today first) ---
    if len(df_secs) >= 2:
        yesterday_all = df_secs.iloc[-2][players].dropna()
        if len(yesterday_all) > 1 and not today_valid.empty:
            yesterday_loser = yesterday_all.idxmax()
            today_winner = today_valid.idxmin()
            if yesterday_loser == today_winner:
                name = alias_map.get(yesterday_loser, yesterday_loser)
                fun_facts.append(
                    f"👑 {name} went from last to first overnight... redemption arc complete! 🔥"
                )

    # --- Exact same time (any two or more players) ---
    if not today_valid.empty:
        for t, count in today_valid.value_counts().items():
            if count >= 2:
                tied = today_valid[today_valid == t].index.tolist()
                names = [alias_map.get(p, p) for p in tied]
                name_str = (
                    ", ".join(names[:-1])
                    + (" and " if len(names) > 1 else "")
                    + names[-1]
                )
                both_or_all = "both" if len(tied) == 2 else "all"
                t_fmt = f"{int(t // 60)}:{int(t % 60):02d}"
                fun_facts.append(
                    f"👀 {name_str} {both_or_all} finished in exactly {t_fmt}... either someone's copying or the matrix glitched 🤖"
                )
                break

    # --- Beat or tied Quincy (always shown when it occurs) ---
    if "Backtracking" in df_secs.columns:
        bt_time = time_to_seconds(df_secs.iloc[-1]["Backtracking"])
        if not pd.isna(bt_time):
            bt_fmt = f"{int(bt_time // 60)}:{int(bt_time % 60):02d}"
            beaters = [
                p for p in players if not pd.isna(today[p]) and today[p] < bt_time
            ]
            tiers = [
                p for p in players if not pd.isna(today[p]) and today[p] == bt_time
            ]
            if beaters:
                names = [alias_map.get(p, p) for p in beaters]
                name_str = (
                    names[0]
                    if len(names) == 1
                    else ", ".join(names[:-1]) + " and " + names[-1]
                )
                if len(beaters) == 1:
                    p_fmt = f"{int(today[beaters[0]] // 60)}:{int(today[beaters[0]] % 60):02d}"
                    return f"🤖 {name_str} solved it in {p_fmt} vs Quincy's {bt_fmt}. Quincy does not accept this result 😤"
                else:
                    return f"🤖 Multiple humans - {name_str} - have beaten Quincy's time of {bt_fmt}. A coordinated uprising... this was not in the training data 📋"
            elif tiers:
                names = [alias_map.get(p, p) for p in tiers]
                name_str = (
                    names[0]
                    if len(names) == 1
                    else ", ".join(names[:-1]) + " and " + names[-1]
                )
                return f"🤖 {name_str} matched Quincy exactly at {bt_fmt}! Not bad for a human... 🤝"

    # --- Pick one fact to show ---
    if fun_facts:
        print(fun_facts)
        return random.choice(fun_facts)
    else:
        return "🤔 No valid stats available for today!"


# Path to the plot image
image_path = r"C:\Users\user\OneDrive\Documents\GitHub\LinkedIn-Queens-Solver\queens_scores_plot.png"


def _send_streak_reminder(driver, missing_players, alias_map):
    if not missing_players:
        return

    textbox = driver.find_element(
        By.XPATH,
        "//div[contains(@class, 'msg-form__contenteditable') and @contenteditable='true']",
    )
    textbox.click()
    time.sleep(0.5)

    for p in missing_players:
        first_name = alias_map.get(p, p)
        ActionChains(driver).send_keys(f"@{first_name}").perform()
        time.sleep(1.5)  # wait for @mention dropdown to appear
        ActionChains(driver).send_keys(Keys.ENTER).send_keys(" ").perform()
        time.sleep(0.3)

    ActionChains(driver).send_keys(
        "you didn't play today! Make sure to play tomorrow to protect your streak! 🎮"
    ).perform()
    time.sleep(0.5)

    actions = ActionChains(driver)
    for _ in range(6):
        actions.send_keys(Keys.TAB).pause(0.2)
    actions.send_keys(Keys.ENTER).perform()
    time.sleep(2)
    print(f"Streak reminder sent to: {[alias_map.get(p, p) for p in missing_players]}")


def upload_plot(driver):
    # 1. Find the image-specific input (accepts image/*)
    upload_input = driver.find_element(
        By.XPATH, "//input[@type='file' and contains(@accept, 'image')]"
    )

    # 2. Upload the file
    upload_input.send_keys(image_path)
    print("📤 Image file sent to upload input.")

    try:
        # Option A: check for generic file preview (not just images)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'msg-attachment-preview')]")
            )
        )
        print("Option A")
    except Exception:
        # Option B: wait for a "Send" button to become active again
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(@class, 'msg-form__send-button')]")
            )
        )
        print("Option B")
    print("🖼️ Image preview appeared in chat.")

    time.sleep(3)

    textbox = driver.find_element(
        By.XPATH,
        "//div[contains(@class, 'msg-form__contenteditable') and @contenteditable='true']",
    )
    textbox.click()
    time.sleep(0.2)

    df = pd.read_csv(
        r"C:\Users\user\OneDrive\Documents\GitHub\LinkedIn-Queens-Solver\Queens\scores.csv"
    )
    fun_fact = get_fun_fact(df)

    actions = ActionChains(driver)
    actions.send_keys(fun_fact).perform()
    time.sleep(0.5)

    # 4. Press TAB then ENTER to send
    actions = ActionChains(driver)

    # Repeat TAB 6 times with a 0.2s pause between each
    for _ in range(6):
        actions.send_keys(Keys.TAB).pause(0.2)

    # Finally, press ENTER
    actions.send_keys(Keys.ENTER).perform()
    print("✅ Image sent via Tab + Enter.")
    time.sleep(5)

    # Streak reminder for anyone who didn't play today
    alias_map = {}
    for key, value in os.environ.items():
        if key.startswith("ALIAS_"):
            first_name = key.replace("ALIAS_", "").split("_")[0].capitalize()
            alias_map[value] = first_name
    players = [p for p in df.columns if p not in ["Day", "DoW", "Backtracking"]]
    today_row = df.iloc[-1]
    missing = [
        p for p in players if pd.isna(today_row[p]) or str(today_row[p]).strip() == ""
    ]
    _send_streak_reminder(driver, missing, alias_map)


if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Users\user\OneDrive\Documents\GitHub\LinkedIn-Queens-Solver\Queens\scores.csv"
    )
    test = get_fun_fact(df)
    print(test)
