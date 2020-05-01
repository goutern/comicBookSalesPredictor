import os
import urllib
from os.path import basename

import requests
import pandas as pd
import html5lib
from bs4 import BeautifulSoup
from numpy import ndenumerate
from tabulate import tabulate
import time
import numpy as np
import lxml
from tempfile import TemporaryFile



def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")

cleandata = np.zeros(shape=(1,8))
for year in range (2000, 2020):
    print(str(year))
    # if not (os.path.exists("comics\\" + str(year))):
    #         os.mkdir("comics\\" + str(year))
    for month in range(1, 13):
        print(str(month))
        # if not (os.path.exists("comics\\"+ str(year)+"\\"+str(month))):
        #     os.mkdir("comics\\"+str(year)+"\\"+str(month))
        doubleDigitMonth = "%02d" % month
        time.sleep(1)
        url=('https://www.comichron.com/monthlycomicssales/' + str(year) + '/'+str(year) + '-' +str(doubleDigitMonth)+'.html')#Create a handle, page, to handle the contents of the website
        page = requests.get(url)#Store the contents of the website under doc

        soup = BeautifulSoup(page.content, 'html.parser')

        tb = soup.findChild('table', class_='comichron-issuetable')
    # rows = tb.findChildren(['tr'])
        df = pd.read_html(str(tb))
        # print( tabulate(df[0], headers='keys', tablefmt='psql') )
        # print(df[0].head())
        comicNames = df[0][['Comic-book Title', 'Issue','Est. units']]
        # print(comicNames)
        comicbooks = comicNames.to_numpy()

        for comicbook in comicbooks:
            filename = make_safe_filename(comicbook[0] + "i" + str(comicbook[1]))
            path = "comics\\"+ str(year) + "\\" + str(month) + "\\" + filename + ".jpg"
            if (os.path.exists(path)):
                comicbook = np.append(comicbook, [path, year, month, comicbook[2],filename])
                if filename in cleandata[:,7]:
                    index = np.where(cleandata[:,[7]] == filename)[0]
                    cleandata[index,6] = int(comicbook[6]) + int(cleandata[index,6])
                else:
                    cleandata = np.vstack((cleandata, comicbook))

cleandata = np.delete(cleandata, 0,0)
namecheck = cleandata[:,7]
u, c = np.unique(namecheck, return_counts=True)
dup = u[c > 1]
print("dupes?: =" + dup)
np.save("cleanData.npy", cleandata)




