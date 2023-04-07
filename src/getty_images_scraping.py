import os
import time
import urllib
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Firefox()
field_type = "rugby"
query = "terrain de rugby"
path = f'./img/{field_type}'
url = "https://www.gettyimages.fr"
driver.get(url)

input_element = driver.find_element(By.XPATH, "/html/body/div/div/header/nav/div[2]/form/div[1]/input")

input_element.send_keys(query)
input_element.submit()

time.sleep(5)

images = driver.find_elements(By.TAG_NAME, "img")

nb_files = len(os.listdir(path))

for index, image in enumerate(images):
    try:
        img_url = image.get_attribute("src")
        img_name = f"{url.split('.')[1].capitalize()}_{field_type}_pitch{index + nb_files + 1}.jpg"
        img_path = os.path.join(path, img_name)
        urllib.request.urlretrieve(img_url, img_path)
    except Exception as e:
        logging.log(logging.INFO, f"Couldn't retrieve this image because {e}")
        continue

logging.log(logging.INFO, f"{query} images downloaded")
driver.quit()
