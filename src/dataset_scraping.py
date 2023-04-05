import os
import urllib
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
field_type = "foot"
query = "soccer field"

site = f"https://www.google.com/imghp?hl=en"
driver.get(site)
button = driver.find_element(By.ID, "W0wltc")
button.click()

input_element = driver.find_element(By.CLASS_NAME, "gLFyf")
input_element.send_keys(query)
input_element.send_keys(Keys.RETURN)

last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

img_elements = driver.find_elements(By.CSS_SELECTOR, ".rg_i")
nb_files = len(os.listdir(f'./img/{field_type}'))

for index, img_element in enumerate(img_elements):
    try:
        img_url = img_element.get_attribute("src")
        img_name = f"foot_pitch{index + nb_files + 1}.jpg"
        img_path = os.path.join('img/foot', img_name)
        urllib.request.urlretrieve(img_url, img_path)
    except:
        pass

driver.quit()
