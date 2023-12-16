import os
import pandas
import requests
import numpy as np
from collections import Counter

# 下载气象数据
def DownloadData(inPath_1, outPath_1):
    for dirPath, dirname, filenames in os.walk(inPath_1):
        for filename in filenames:
            inPath_2 = os.path.join(dirPath, filename)
            file_1 = open(inPath_2)
            for line in file_1:
                filename_1 = line.split('/')[12].split('?')[0]
                outPath_2 = os.path.join(outPath_1, filename_1)
                if os.path.exists(outPath_2):
                    pass
                else:
                    download_1 = requests.get(line)
                    with open(outPath_2, 'wb') as code:
                         code.write(download_1.content)
                    print(filename_1 + ' is downloaded!')
            print(filename + ' is downloaded!')

# 检查下载是否完整
def CheckDownload(inPath_1):
    for year_1 in range(1987, 2020):
        for month_1 in range(1, 13):
            time_1 = str(year_1) + str(month_1).zfill(2) + '.TXT'
            filename_1 = 'SURF_CLI_CHN_MUL_DAY-EVP-13240-' + time_1
            inPath_2 = os.path.join(inPath_1, filename_1)
            if os.path.exists(inPath_2):
                pass
            else:
                print(time_1)

# 只保留有连续记录的监测数据
def SelectRecords(inPath_1, outPath_1):
    stationIDList_1 = []
    for dirPath, dirname, filenames in os.walk(inPath_1):
        for filename in filenames:
            if 'SURF_CLI_CHN_MUL_DAY-TEM' in filename:
                inPath_2 = os.path.join(dirPath, filename)
                txt_1 = np.loadtxt(inPath_2)
                dataFrame_1 = pandas.DataFrame(txt_1)
                stationIDList_2 = dataFrame_1[0].unique()
                for stationID_1 in stationIDList_2:
                    stationIDList_1.append(stationID_1)
    stationIDList_3 = []
    counter_1 = Counter(stationIDList_1)
    for item_1 in counter_1.items():
        if item_1[1] == 408:
            stationIDList_3.append(item_1[0])
    print(stationIDList_3)
    for dirPath, dirname, filenames in os.walk(inPath_1):
        for filename in filenames:
            outPath_2 = os.path.join(outPath_1, filename.replace('.TXT', '.csv', 1))
            if os.path.exists(outPath_2):
                pass
            else:
                inPath_3 = os.path.join(dirPath, filename)
                txt_2 = np.loadtxt(inPath_3)
                dataFrame_2 = pandas.DataFrame(txt_2)
                dataFrame_3 = dataFrame_2[dataFrame_2[0].isin(stationIDList_3)]
                dataFrame_3.to_csv(outPath_2, index = False)

# 只保留有连续记录的监测数据
def MergeRecords(inPath_1, outPath_1, variable_1):
    dataFrame_1 = pandas.DataFrame()
    outPath_2 = os.path.join(outPath_1, variable_1 + '.csv')
    if os.path.exists(outPath_2):
        pass
    else:
        for dirPath, dirname, filenames in os.walk(inPath_1):
            for filename in filenames:
                if variable_1 in filename:
                    inPath_2 = os.path.join(inPath_1, filename)
                    dataFrame_2 = pandas.read_csv(inPath_2)
                    dataFrame_1 = pandas.concat([dataFrame_1, dataFrame_2])
        dataFrame_1.to_csv(outPath_2, index = False)
        print(variable_1 + ' is done!')

    # for year_1 in range(1990, 2020):
    #     dataFrame_1 = pandas.DataFrame()
    #     outPath_2 = os.path.join(outPath_1, variable_1 + '_' + str(year_1) + '.csv')
    #     if os.path.exists(outPath_2):
    #         pass
    #     else:
    #         for dirPath, dirname, filenames in os.walk(inPath_1):
    #             for filename in filenames:
    #                 if str(year_1) in filename and variable_1 in filename:
    #                     inPath_2 = os.path.join(inPath_1, filename)
    #                     dataFrame_2 = pandas.read_csv(inPath_2)
    #                     dataFrame_1 = pandas.concat([dataFrame_1, dataFrame_2])
    #         dataFrame_1.to_csv(outPath_2, index = False)
    #         print(str(year_1) + ' is done!')

# 获取站点信息
def GetStations(inPath_1, outPath_1):
    dataFrame_1 = pandas.read_csv(inPath_1)
    stationIDList_1 = dataFrame_1['0'].unique()
    latitudeList_1 = []
    longitudeList_1 = []
    altitudeList_1 = []
    for stationID in stationIDList_1:
       dataFrame_2 = dataFrame_1[dataFrame_1['0'] == stationID]
       latitude_1 = dataFrame_2['1'].unique()[0]
       longitude_1 = dataFrame_2['2'].unique()[0]
       altitude_1 = dataFrame_2['3'].unique()[0]
       latitudeList_1.append(latitude_1)
       longitudeList_1.append(longitude_1)
       altitudeList_1.append(altitude_1)
    stationIDArray_1 = np.array(stationIDList_1)
    latitudeArray_1 = np.array(latitudeList_1)
    longitudeArray_1 = np.array(longitudeList_1)
    altitudeArray_1 = np.array(altitudeList_1)
    dataFrame_3 = pandas.DataFrame({'stationID': stationIDArray_1, 'latitude': latitudeArray_1, 'longitude': longitudeArray_1, 'altitude': altitudeArray_1})
    dataFrame_3.to_csv(outPath_1, index = False)


def main():
    # # 下载气象数据
    # inPath_1 = r'F:\2_master_paper\data\5_observation\1_link'
    # outPath_1 = r'F:\2_master_paper\data\5_observation\2_download'
    # DownloadData(inPath_1, outPath_1)
    
    # # 检查下载是否完整
    # inPath_1 = r'F:\2_master_paper\data\5_observation\2_download'
    # CheckDownload(inPath_1)

    # 只保留有连续记录的监测数据
    inPath_1 = r'F:\2_master_paper\data\5_observation\2_download'
    outPath_1 = r'F:\2_master_paper\data\5_observation\3_select'
    SelectRecords(inPath_1, outPath_1)

    # # 获取站点信息
    # inPath_1 = r'F:\2_master_paper\data\5_observation\0_base\station_1.csv'
    # outPath_1 = r'F:\2_master_paper\data\5_observation\0_base\station_2.csv'
    # GetStations(inPath_1, outPath_1)

    # # 只保留有连续记录的监测数据
    # inPath_1 = r'F:\2_master_paper\data\5_observation\3_select'
    # outPath_1 = r'F:\2_master_paper\data\5_observation\4_merge'
    # variable_1 = 'EVP'
    # MergeRecords(inPath_1, outPath_1, variable_1)

main()
