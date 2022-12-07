# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.
import os
import glob
import datetime
import cv2 as cv
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
import pytesseract
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from collections import Counter


def check_text_is_numeric(text):
    # ha nem szám
    if not text.strip().isnumeric():
        text = "NaN"
    else:
        # ha szám, de több karakterből áll
        if len(text.strip()) != 1:
            text = "NaN"
        # ha szám akkor legyen int
        else:
            text = int(text)
    return text


# saját mask
def get_digit_new_mask_technique(img):
    mask_h = round(img.shape[0] * 1, 15)
    mask_w = round(img.shape[1] * 1, 15)
    mask = np.zeros((mask_h, mask_w, 3), dtype="uint8")
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    h, w = img.shape
    hh, ww = mask.shape
    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)
    masked_img = mask.copy()
    masked_img[yoff:yoff + h, xoff:xoff + w] = img
    text = (pytesseract.image_to_string(masked_img, config="--psm 10 outputbase digits"))
    text = check_text_is_numeric(text)
    return text


# kiszedem az NaN-kat, majd megkeresem a legtöbbet szereplőt
# Ha csak NaN volt benne akkor az lesz a legtöbbet szereplő
def get_most_common_element(iList):
    temp_list = list(filter(("NaN").__ne__, iList))
    most_common_digit = [word for word, word_count in Counter(temp_list).most_common(1)]
    if len(most_common_digit) == 0: most_common_digit.append("NaN")
    return most_common_digit[0]


# Based on Original
def get_digit_based_on_original_img(img, col_name, y_start, y_end, x_start, x_end):
    digit_list = []
    for i in range(5):
        x_start = x_start - i
        y_start = y_start - i
        x_end = x_end + i
        y_end = y_end + i
        temp_img = img[y_start:y_end, x_start:x_end]
        text = (pytesseract.image_to_string(temp_img, config="--psm 10 outputbase digits"))
        text = check_text_is_numeric(text)
        if text == "NaN": text = get_digit_new_mask_technique(img)
        digit_list.append(text)
    text = get_most_common_element(digit_list)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# GaussianBlur alapján
def get_digit_median_blur(img, col_name):
    img = cv.medianBlur(img, 5)
    text = (pytesseract.image_to_string(img, config="--psm 10 outputbase digits"))
    text = check_text_is_numeric(text)
    if text == "NaN": text = get_digit_new_mask_technique(img)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# Simple Thresholding
def get_digit_simple_th(img, col_name):
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    text = (pytesseract.image_to_string(th1, config="--psm 10 outputbase digits"))
    text = check_text_is_numeric(text)
    if text == "NaN": text = get_digit_new_mask_technique(th1)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# 'Adaptive Mean Thresholding'
def get_digit_adaptive_mean_th(img, col_name):
    digit_list = []
    for img_variation in range(2):
        #if img_variation == 1: img = cv.medianBlur(img, 5)
        for block_size in [11, 13, 17, 19, 23, 29, 31]:
            for c in range(1, 4):
                th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                           cv.THRESH_BINARY, block_size, c)
                text = (pytesseract.image_to_string(th2, config="--psm 10 outputbase digits"))
                text = check_text_is_numeric(text)
                if text == "NaN": text = get_digit_new_mask_technique(th2)
                digit_list.append(text)
    text = get_most_common_element(digit_list)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# 'Adaptive Gaussian Thresholding'
def get_digit_adaptive_gaussian_th(img, col_name):
    digit_list = []
    for img_variation in range(2):
        #if img_variation == 1: img = cv.medianBlur(img, 5)
        for block_size in [11, 13, 17, 19, 23, 29, 31]:
            for c in range(1, 4):
                th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                           cv.THRESH_BINARY, block_size, c)
                text = (pytesseract.image_to_string(th3, config="--psm 10 outputbase digits"))
                text = check_text_is_numeric(text)
                if text == "NaN": text = get_digit_new_mask_technique(th3)
                digit_list.append(text)
    text = get_most_common_element(digit_list)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# inRange Thresholding
def get_digit_in_range_th(path, col_name, y_start, y_end, x_start, x_end):
    img = cv.imread(path)
    img = img[y_start:y_end, x_start:x_end]
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_image, array([0, 0, 0]), array([255, 255, 255]))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 3))
    dilate = cv.dilate(mask, kernel, iterations=1)
    inRangeTh = cv.bitwise_and(dilate, mask)
    text = (pytesseract.image_to_string(inRangeTh, config="--psm 10 outputbase digits"))
    text = check_text_is_numeric(text)
    if text == "NaN": text = get_digit_new_mask_technique(dilate)
    temp_df = pd.DataFrame([{col_name: text}])
    return temp_df


# choose the most appropriate digit
def choose_digit(input_df, input_col_name, opt_list=[]):
    # opt_list: List which contains the current measurement digits
    # BOO = Based On Original
    # If list was provided take into consideration its last digit
    if len(opt_list) != 0:
        # If digit was found BOO image that will be used
        if isinstance(input_df[input_col_name].iat[0], int) and input_df[input_col_name].iat[0] >= opt_list[-1] and (input_df[input_col_name].iat[0] - opt_list[-1]) < 3:
            if input_df[input_col_name].iat[0] - opt_list[-1] < 4:
                return input_df[input_col_name].iat[0]
        # If BOO was found and it is 0 and
        elif isinstance(input_df[input_col_name].iat[0], int) and input_df[input_col_name].iat[0] == 0 and opt_list[
            -1] == 9:
            return input_df[input_col_name].iat[0]
        # If th2 and th3 are the same use that
        elif isinstance(input_df[input_col_name].iat[3], int) and input_df[input_col_name].iat[3] == \
                input_df[input_col_name].iat[4]:
            if input_df[input_col_name].iat[3] - opt_list[-1] < 4:
                return input_df[input_col_name].iat[3]
            elif opt_list[-1] == 9 and input_df[input_col_name].iat[3] == 0:
                return input_df[input_col_name].iat[3]
        # Filter out "NaN"
        temp_list = list(filter(("NaN").__ne__, input_df[input_col_name].tolist()))
        if opt_list[-1] != 9:
            # Find the numbers which are bigger than the original value but not more with 3
            filtered_list = filter(lambda digit: abs(digit - opt_list[-1]) <= 2, temp_list)
            return get_most_common_element(filtered_list)
        # Find the closest value to 0
        else:
            return min(temp_list, key=lambda x: abs(x - 0))
    else:
        if isinstance(input_df[input_col_name].iat[0], int):
            return input_df[input_col_name].iat[0]
        # If th2 and th3 are the same use that
        elif isinstance(input_df[input_col_name].iat[3], int) and input_df[input_col_name].iat[3] == \
                input_df[input_col_name].iat[4]:
            return input_df[input_col_name].iat[3]
        else:
            return get_most_common_element(input_df[input_col_name].tolist())


def update_value_list(chosen_value, value_list):
    # Ha a talált szám NaN akkor nem változtatok az eredeti értéken
    # Ha nem NaN és a talált érték nagyobb akkor megváltoztatom a talált értékre
    if chosen_value == "NaN":
        return value_list
    elif value_list[-1] < chosen_value:
        value_list[5] = chosen_value
    # Ha 9 volt a tárolt és 0 a talált akkor növelni kell a többi karakter értékét annak függvényében
    # hogy azok értéke nem 9 volt e
    elif value_list[5] == 9 and chosen_value == 0:
        value_list[5] = chosen_value
        if value_list[4] == 9:
            value_list[4] = 0
            if value_list[3] == 9:
                value_list[3] = 0
                if value_list[2] == 9:
                    value_list[2] = 0
                    if value_list[1] == 9:
                        value_list[1] = 0
                        value_list[0] = value_list[0] + 1
                    else:
                        value_list[1] = value_list[1] + 1
                else:
                    value_list[2] = value_list[2] + 1
            else:
                value_list[3] = value_list[3] + 1
        else:
            value_list[4] = value_list[4] + 1
    return value_list


def create_final_value(input_list):
    # Create a number from the list's digits
    for (elem, counter) in zip(input_list, range(6)):
        if counter == 0:
            output_value = str(elem)
        elif counter == 5:
            output_value = output_value + '.' + str(elem)
        else:
            output_value = output_value + str(elem)
    return output_value


def get_results(directory, start_value_df, results_df):
    # Az Excelben megadott kezdő érték listába írása
    measurement_digits_list = start_value_df.loc[0].to_list()

    #Check if there is already data or empty
    if results_df.iloc[-1].to_list()[2] != "":
        temp_for_digits = results_df.iloc[-1].to_list()[2]
        measurement_digits_list.clear()
        for digit in str(temp_for_digits):
            if digit != ".":
                measurement_digits_list.append(int(digit))

    # Loop through the directory (images)
    #for image in os.listdir(directory):
    # Get tha image's file path
    image = directory
    path = os.path.join(directory, image)
    image = image.split("\\")[-1]
    # Get date and time from file name
    date = (image.split("_")[0]).replace("-", ".")
    time = ((image.split("_")[1]).split(".")[0]).replace("-", ":")

    # Open the image
    img = cv.imread(path, 0)

    # Based on Original
    df_col = get_digit_based_on_original_img(img, col_name, y_start, y_end, x_start, x_end)

    # crop img
    img = img[y_start:y_end, x_start:x_end]

    # GaussianBlur
    temp_df = get_digit_median_blur(img, col_name)
    df_col = pd.concat([df_col, temp_df])

    # Simple Thresholding
    temp_df = get_digit_simple_th(img, col_name)
    df_col = pd.concat([df_col, temp_df])

    # 'Adaptive Mean Thresholding'
    temp_df = get_digit_adaptive_mean_th(img, col_name)
    df_col = pd.concat([df_col, temp_df])

    # 'Adaptive Gaussian Thresholding'
    temp_df = get_digit_adaptive_gaussian_th(img, col_name)
    df_col = pd.concat([df_col, temp_df])

    # inRange Thresholding
    temp_df = get_digit_in_range_th(path, col_name, y_start, y_end, x_start, x_end)
    df_col = pd.concat([df_col, temp_df])

    # choose the most appropriate digit
    chosen_digit = choose_digit(df_col, col_name, measurement_digits_list)

    # update the previous digits in the list
    measurement_digits_list = update_value_list(chosen_digit, measurement_digits_list)

    # Create a number from the list's elements
    final_value = create_final_value(measurement_digits_list)

    # clear dataframe
    df_col = df_col.iloc[0:0]

    # append the new measurement
    results_df = pd.concat([results_df, pd.DataFrame({
        'date': [date],
        'time': [time],
        'value': [float(final_value)],
        'file_path': [path]})])

    print("get_results: finished")

    return results_df


def get_detailed_results(directory, detailed_results_df):
    counter = 1
    #for image in os.listdir(directory):
    image = directory
    path = os.path.join(directory, image)
    image = image.split("\\")[-1]
    date = (image.split("_")[0]).replace("-", ".")
    time = ((image.split("_")[1]).split(".")[0]).replace("-", ":")
    for (x_start, y_start, x_end, y_end, col_name) in zip(x_start_list, y_start_list, x_end_list, y_end_list,
                                                          col_name_list):
        img = cv.imread(path, 0)

        # Based on Original
        df_col = get_digit_based_on_original_img(img, col_name, y_start, y_end, x_start, x_end)

        # crop img
        img = img[y_start:y_end, x_start:x_end]

        # GaussianBlur alapján
        temp_df = get_digit_median_blur(img, col_name)
        df_col = pd.concat([df_col, temp_df])

        # Simple Thresholding
        temp_df = get_digit_simple_th(img, col_name)
        df_col = pd.concat([df_col, temp_df])

        # 'Adaptive Mean Thresholding'
        temp_df = get_digit_adaptive_mean_th(img, col_name)
        df_col = pd.concat([df_col, temp_df])

        # 'Adaptive Gaussian Thresholding'
        temp_df = get_digit_adaptive_gaussian_th(img, col_name)
        df_col = pd.concat([df_col, temp_df])

        # inRange Thresholding
        temp_df = get_digit_in_range_th(path, col_name, y_start, y_end, x_start, x_end)
        df_col = pd.concat([df_col, temp_df])

        # choose the most appropriate digit
        chosen_digit = choose_digit(df_col, col_name)
        temp_df = pd.DataFrame([{col_name: chosen_digit}])
        df_col = pd.concat([df_col, temp_df])

        # put all the (dataframe) columns into one dataframe
        df_all_temp = df_col
        if col_name != "fourth":
            df_all = pd.concat([df_all, df_all_temp], axis=1)
        else:
            df_all = df_col

        # clear dataframe
        df_col = df_col.iloc[0:0]

    df_all = df_all.reset_index()

    # Create the other neccessary (static) columns for the output file
    df_other_cols = pd.DataFrame(
        {'method': ["From Original", "GaussianBlur", "Simple Thresholding", "Adaptive Mean Thresholding",
                    "Adaptive Gaussian Thresholding", "inRange Thresholding", "Final value"],
         'first': [1, 1, 1, 1, 1, 1, 1],
         'second': [1, 1, 1, 1, 1, 1, 1],
         'third': [0, 0, 0, 0, 0, 0, 0],
         'file_path': [path, path, path, path, path, path, path],
         'date': [date, date, date, date, date, date, date],
         'time': [time, time, time, time, time, time, time]})

    # Concatenate the measurements with the static columns
    df_all = pd.concat([df_all, df_other_cols], axis=1)

    # Concatenate the new and the old data
    detailed_results_df = pd.concat([detailed_results_df, df_all])
    detailed_results_df = detailed_results_df.drop(columns=['index'])

    # Count the opened images
    counter += 1

    # clear dataframe
    df_all = df_all.iloc[0:0]

    print("get_detailed_results: finished")

    return detailed_results_df


if __name__ == '__main__':
    now = datetime.datetime.now()
    print(now)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

    x_start = 1254
    y_start = 595
    x_end = 1288
    y_end = 665

    x_start_list = [1100, 1180, 1255, 1342, 1430]
    y_start_list = [630, 622, 620, 620, 608]
    x_end_list = [1140, 1219, 1300, 1382, 1475]
    y_end_list = [695, 690, 686, 680, 684]

    col_name = "6th digits"
    col_name_list = ["fourth", "fifth", "sixth", "seventh", "eighth"]

    directory_path = r"C:\Users\bmoli\Desktop\test_pcs\fifty"
    excel_path = r"C:\Users\bmoli\PycharmProjects\GasMeasurement\Gas_Measurements.xlsx"
    results_df = pd.read_excel(excel_path, sheet_name='Results')
    start_value_df = pd.read_excel(excel_path, sheet_name='Setup')
    detailed_results_df = pd.read_excel(excel_path, sheet_name='Detailed_Results')

    list_of_files = glob.glob(r"\\LAPTOP-NS42NFQM\grabs\*")  # * means all if need specific format then *.csv
    #list_of_files = glob.glob(r"C:\Users\bmoli\Desktop\test_pcs\fifty\*")
    latest_file = max(list_of_files, key=os.path.getctime)

    results_df = get_results(latest_file, start_value_df, results_df)

    now = datetime.datetime.now()
    print(now)

#    detailed_results_df = get_detailed_results(latest_file, detailed_results_df)

    # write the new data to the Excel file
    with pd.ExcelWriter(excel_path) as writer:
        results_df.to_excel(writer, sheet_name='Results', index=False)
        start_value_df.to_excel(writer, sheet_name='Setup', index=False)
        detailed_results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

    now = datetime.datetime.now()
    print(now)
    print("all finished")
