import requests
import pandas as pd
import html5lib
from bs4 import BeautifulSoup
from tabulate import tabulate

for year in range (2000, 2020):
    for month in range(11, 13):
        doubleDigitMonth = "%02d" % month
        url=('https://www.comichron.com/monthlycomicssales/' + str(year) + '/'+str(year) + '-' +str(doubleDigitMonth)+'.html')#Create a handle, page, to handle the contents of the website
        page = requests.get(url)#Store the contents of the website under doc
    # print(page.status_code)
    # print(page.content)

        soup = BeautifulSoup(page.content, 'html.parser')
    # print(soup.prettify())

        tb = soup.findChild('table',class_='comichron-issuetable')
    # rows = tb.findChildren(['tr'])
        df = pd.read_html(str(tb))
        print( tabulate(df[0], headers='keys', tablefmt='psql') )
    # for row in rows:
    #     cells = row.findChildren(['td','th'])
    #     for cell in cells:
    #         value = cell.string
    #         print ("The value in this cell is %s" % value)
