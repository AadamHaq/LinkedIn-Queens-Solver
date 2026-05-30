import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# The results overlay and all share modals live inside the interop iframe,

_IFRAME = '[data-testid="interop-iframe"]'


def _switch_to_iframe(driver):
    iframe = driver.find_element(By.CSS_SELECTOR, _IFRAME)
    driver.switch_to.frame(iframe)


def share_score(driver, name):
    """
    Function: Shares Queens score to a group chat

    Args:
        driver: Selenium driver that was initialised
        name: Name of group chat that the score will be sent to

    Description: Switches into the LinkedIn interop iframe, waits for the
    results screen, then shares the score to the named group chat.
    """

    try:
        _switch_to_iframe(driver)
        print("Switched into interop iframe")
    except Exception as e:
        print(f"Could not switch to iframe: {e}")
        return

    print("Waiting for results screen...")
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".pr-game-results__components")
            )
        )
        print("Results screen detected")
    except TimeoutException:
        print("Results screen did not load in time")
        driver.switch_to.default_content()
        return

    time.sleep(1)

    # Dismiss streak-freeze / bonus popup if present
    try:
        skip_btn = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[normalize-space()='Skip' or normalize-space()='No thanks' or normalize-space()='Decline' or normalize-space()='Not now']",
                )
            )
        )
        skip_btn.click()
        print("Bonus popup closed")
        time.sleep(1)
    except TimeoutException:
        print("No bonus popup present")

    # Click the Send button.
    # The button has no text — its label is a sibling <span class="pr-top__text t-14">Send</span>.
    # Find it via its SVG icon attribute.
    print("Looking for Send button...")
    try:
        send_btn = WebDriverWait(driver, 10).until(
            lambda d: d.execute_script(
                "const svg=document.querySelector('[data-test-icon=\"send-privately-medium\"]');"
                "return svg?svg.closest('button'):null;"
            )
        )
        send_btn.click()
        print("Clicked Send button")
        time.sleep(3.5)
    except TimeoutException:
        print("Could not find Send button")
        try:
            driver.save_screenshot("debug_send_button.png")
        except Exception:
            pass
        driver.switch_to.default_content()
        return

    # Select recipient — try pre-populated list first, then type-ahead
    send_to = None
    try:
        send_to = WebDriverWait(driver, 5).until(
            lambda d: d.execute_script(
                "for(const d of document.querySelectorAll('.msg-connections-typeahead__entity-description')){"
                "  const dt=d.querySelector('dt');"
                "  if(dt&&dt.textContent.includes(arguments[0])){"
                "    let el=d;while(el&&el.tagName!=='BUTTON')el=el.parentElement;return el;"
                "  }}"
                "return null;",
                name,
            )
        )
    except TimeoutException:
        pass

    if send_to:
        driver.execute_script("arguments[0].click()", send_to)
        print(f"Selected recipient: {name}")
    else:
        input_box = None
        try:
            input_box = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "input[placeholder='Type a name']")
                )
            )
        except TimeoutException:
            pass

        if not input_box:
            print(f"Could not find recipient input for: {name}")
            driver.switch_to.default_content()
            return

        input_box.send_keys(name)
        time.sleep(1)

        result = driver.execute_script(
            "for(const d of document.querySelectorAll('.msg-connections-typeahead__entity-description')){"
            "  const dt=d.querySelector('dt');"
            "  if(dt&&dt.textContent.includes(arguments[0])){"
            "    const row=d.closest('[class*=\"search-result-row\"]');"
            "    if(row){const lbl=row.querySelector('[class*=\"checkbox\"]');if(lbl)return lbl;}"
            "    return d;"
            "  }}"
            "return null;",
            name,
        )
        if result:
            driver.execute_script("arguments[0].click()", result)
            print(f"Typed and selected recipient: {name}")
        else:
            print(f"Could not find recipient: {name}")
            driver.switch_to.default_content()
            return

    # Send the message
    time.sleep(0.5)
    try:
        textbox = driver.find_element(
            By.CSS_SELECTOR, ".msg-form__contenteditable[contenteditable='true']"
        )
        driver.execute_script("arguments[0].focus()", textbox)
        time.sleep(0.2)
    except Exception:
        pass

    ActionChains(driver).send_keys(Keys.TAB).pause(0.2).send_keys(Keys.ENTER).perform()
    print("Message sent successfully")
    time.sleep(2)

    driver.switch_to.default_content()
