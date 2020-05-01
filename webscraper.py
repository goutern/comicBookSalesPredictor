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
import lxml

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")

for year in range (2000, 2020):
    print(str(year))
    if not (os.path.exists("comics\\" + str(year))):
            os.mkdir("comics\\" + str(year))
    for month in range(1, 13):
        print(str(month))
        if not (os.path.exists("comics\\" + str(year)+"\\"+str(month))):
            os.mkdir("comics\\" + str(year)+"\\"+str(month))
        doubleDigitMonth = "%02d" % month
        time.sleep(1)
        url=('https://www.comichron.com/monthlycomicssales/' + str(year) + '/'+str(year) + '-' +str(doubleDigitMonth)+'.html')#Create a handle, page, to handle the contents of the website
        page = requests.get(url)#Store the contents of the website under doc

        soup = BeautifulSoup(page.content, 'html.parser')

        tb = soup.findChild('table',class_='comichron-issuetable')
    # rows = tb.findChildren(['tr'])
        df = pd.read_html(str(tb))
        print( tabulate(df[0], headers='keys', tablefmt='psql') )
        # print(df[0].head())
        comicNames = df[0][['Comic-book Title', 'Issue']]
        # print(comicNames)
        comicbooks = comicNames.to_numpy()
        index = 0
        for comicbook in comicbooks:
            filename = make_safe_filename(comicbook[0] + "i" + str(comicbook[1]))
            path = "comics\\" + str(year) + "\\" + str(month) + "\\" + filename + ".jpg"
            if(os.path.exists(path)):
                continue
            try:
                url = ('https://www.comics.org/searchNew/?q=\"' + comicbook[0] +" "+ comicbook[1] +"\"&search_object=issue&sort=alpha")
            except:
                print("Failed to get: " + comicbook[0])
                continue
            # print(url)
            page = requests.get(url)  # Store the contents of the website under doc
        # print(page.status_code)
        # print(page.content)

            soup = BeautifulSoup(page.content, 'html.parser')
        # print(soup.prettify())

            tb = soup.findChild('table', class_='listing left')
            try:
                rows = tb.findChildren(['tr'])
            except:
                print("Failed to get: " + comicbook[0])
                continue
        # print(rows)

            url = ""
            for link in rows[1].find_all('a'):
                # print(link.get('href'))
                url = 'https://www.comics.org' + link.get('href')

            # print(url)
            page = requests.get(url)  # Store the contents of the website under doc
        # print(page.status_code)
        # print(page.content)

            soup = BeautifulSoup(page.content, 'html.parser')

            productDivs = soup.findAll('div', attrs={'class': 'coverImage'})

            for div in productDivs:
                print(div.find('a')['href'])
                coverPage = 'https://www.comics.org' + div.find('a')['href']
                try:
                    page = requests.get(coverPage)
                except:
                    print("Failed to get: " + comicbook[0])
                    continue
                soup = BeautifulSoup(page.content, 'html.parser')
                coverDivs = soup.findAll('div', attrs={'class': 'issue_covers'})
                for cover in coverDivs:
                    for link in cover.select("img[src^=http]"):
                        lnk = link["src"]
                        try:
                            urllib.request.urlretrieve(lnk,path)
                            # print(comicbook[0])
                        except:
                            print("Failed to get: " + comicbook[0])
                        # time.sleep(1)


# for row in rows:
    #     cells = row.findChildren(['td','th'])
    #     for cell in cells:
    #         value = cell.string
    #         print ("The value in this cell is %s" % value)

# url = ('https://www.comics.org/searchNew/?q=%22' + "avengers%201959"  +  "%20" + "3" + " %22")
# page = requests.get(url)  # Store the contents of the website under doc
# # print(page.status_code)
# # print(page.content)
#
# soup = BeautifulSoup(page.content, 'html.parser')
# # print(soup.prettify())
#
# tb =  soup.findChild('table',class_='listing left')
# rows = tb.findChildren(['tr'])
# # print(rows)
#
# url = ""
# for link in rows[1].find_all('a'):
#     print(link.get('href'))
#     url = 'https://www.comics.org' + link.get('href')
#
# print(url)
# page = requests.get(url)  # Store the contents of the website under doc
# # print(page.status_code)
# # print(page.content)
#
# soup = BeautifulSoup(page.content, 'html.parser')
#
# productDivs = soup.findAll('div', attrs={'class' : 'coverImage'})
#
# for div in productDivs:
#     print(div.find('a')['href'])
#     coverPage = 'https://www.comics.org' + div.find('a')['href']
#     page = requests.get(coverPage)
#     soup = BeautifulSoup(page.content, 'html.parser')
#     coverDivs = soup.findAll('div', attrs={'class': 'issue_covers'})
#     for cover in coverDivs:
#         for link in cover.select("img[src^=http]"):
#             lnk = link["src"]
#             urllib.request.urlretrieve(lnk, basename("avengers.jpg"))
#
#         # print(cover.find(('img')['src']))
#         # lnk = cover.find(('img'['src']))
#         # print(lnk)
#         # with open(basename(lnk[0]), "wb") as f:
#         #     f.write("avengers.jpg")
#
#
#
# # cells = rows[1].findAll('td')
# # for cell in cells:
# #     href = cell.a
# #     # href = ((str)href).split('"')[1::2]
# #     print(href)
# # link = cells.get('href')
# # for link in soup.select('td.embedded_flag a[href]'):
# #     print(link['href'])
# # print(cells)
# # df = pd.read_html(str(tb))
# # print( tabulate(df[0], headers='keys', tablefmt='psql') )