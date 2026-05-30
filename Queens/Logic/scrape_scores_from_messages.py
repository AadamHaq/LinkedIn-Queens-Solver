import re
import time

from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def scroll_chat(driver, key_presses=132, pause=0.03):
    """
    Function: Scrolls up the LinkedIn group chat to load more messages

    Args:
        driver: Selenium driver that was initialised
        key_presses: Number of times Up Key is pressed. Default is 200
        pause: Seconds between each key press

    Description: Scrolls up using up keys after pressing within the chat container
    """
    for attempt in range(3):  # Multiple attempts as it sometimes doesn't work
        try:
            bubbles = driver.find_elements(
                By.CSS_SELECTOR, ".msg-s-event-listitem__message-bubble"
            )  # Find the inside of the chatbox
            if not bubbles:
                print("No message bubbles found to scroll.")
                time.sleep(1)
                continue
            target = bubbles[-1]  # Latest message
            driver.execute_script("arguments[0].scrollIntoView(true);", target)
            time.sleep(0.3)

            offset_x, offset_y = 100, 10
            ActionChains(driver).move_to_element_with_offset(
                target, offset_x, offset_y
            ).click().perform()
            time.sleep(0.3)

            for i in range(1, key_presses + 1):
                target.send_keys(Keys.ARROW_UP)
                time.sleep(pause)

                # Every multiple of 33, send 2 Down Arrows
                if i % 33 == 0:
                    target.send_keys(Keys.ARROW_DOWN)
                    time.sleep(pause)
                    target.send_keys(Keys.ARROW_DOWN)
                    time.sleep(pause)

            print(
                f"Scrolled chat by sending {key_presses} UP arrow keys with 2 DOWN arrows every 33 presses."
            )
            return True

        except (StaleElementReferenceException, NoSuchElementException):
            print(f"Attempt {attempt + 1}: Encountered stale element, retrying...")
            time.sleep(1)
    print("Failed to scroll chat by keys after retries.")
    return False


def get_chat_html(driver):
    """
    Function: Gets the html for beautiful soup

    Args:
        driver: Selenium driver that was initialised

    Description: Finds the messages list. Created as a seperate function for debugging.
    """
    # First path should work but second is a fallback when scrolling to the top of the container
    xpath_primary = (
        "//div[contains(@class,'msg-s-message-list') and contains(@class,'scrollable')]"
    )
    xpath_fallback = "//div[contains(@class,'msg-s-message-list-content') and contains(@class,'list-style-none')]"

    for xpath in [xpath_primary, xpath_fallback]:
        try:
            container = driver.find_element(By.XPATH, xpath)
            return container.get_attribute("innerHTML")
        except NoSuchElementException:
            continue
    raise RuntimeError("Chat container not found")


def scrape_messages_bs4(driver):
    """
    Function: Scrapes Queens messages with Beautiful Soup 4

    Args:
        driver: Selenium driver that was initialised

    Description: Scrapes all messages that starts as `Queens #ddd` where ddd are digits
    """
    # Scroll chat using arrow keys to load all messages
    scrolled = scroll_chat(driver)
    if not scrolled:
        print("Scrolling failed, continuing with current loaded messages.")
    time.sleep(1.5)  # Wait for lazy load

    # Get html messages
    html = get_chat_html(driver)
    soup = BeautifulSoup(html, "html.parser")
    lis = soup.select(
        "li.msg-s-message-list__event"
    )  # Messages are all stored in their own <li>

    # Find 'Today' divider as we are only looking at scores Today
    start = 0
    for i, li in enumerate(lis):
        t = li.find("time", class_="msg-s-message-list__time-heading")
        if t and t.get_text(strip=True) == "Today":
            start = i
            break

    # Now we have 'Today' we can find all messages underneath
    results = []
    current_sender = "Unknown"  # Store sender seperate and track as sender isn't necessarily in the same <li> as the message
    for li in lis[start:]:
        meta = li.select_one(".msg-s-message-group__meta")
        if meta:
            nm = meta.select_one(".msg-s-message-group__name")  # Name
            if nm and nm.get_text(strip=True):
                current_sender = nm.get_text(strip=True)

        bubble = li.select_one(".msg-s-event-listitem__body")  # Message
        if not bubble:
            continue
        text = bubble.get_text("\n", strip=True)  # Strip blank space
        first = next(
            (line for line in text.splitlines() if line.strip()), ""
        )  # Strip new lines
        if re.match(r"^Queens #(\d{3})(?!\d)\b", first):  # Find Queens message
            results.append((current_sender, text))  # Append

    return results


def score_scraper(driver, name):
    """
    Function: Scrapes all scores from the Group Chat

    Args:
        driver: Selenium driver that was initialised
        name: Name of group chat that the score will be sent to

    Description: Combines previous functions and scrapes all scores
    """
    driver.get("https://www.linkedin.com/messaging/")
    time.sleep(3)

    search_bar = driver.find_element(
        By.ID, "search-conversations"
    )  # Enter conversation name
    search_bar.send_keys(name + Keys.ENTER)

    try:
        time.sleep(2)
        conversation = driver.find_element(
            By.XPATH,
            f"//span[text()='{name}']/ancestor::div[contains(@class, 'msg-conversation-card__content--selectable')]",
        )
        conversation.click()  # Click conversation

        time.sleep(2)
        results = scrape_messages_bs4(driver)  # Scrape
        return results
    except Exception as e:
        print(f"Could not find button for {name}: {e}")
