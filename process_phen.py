# -*- coding: utf-8 -*-
import os

import numpy as np
import netCDF4 as nc
from osgeo import gdal, osr, ogr
import glob
import matplotlib.pyplot as plt
import pandas
import scipy.stats
import datetime
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# import arcpy
# from arcpy import env
# from arcpy.sa import *

# 创建文件夹
def CreateDir(inPath_1):
	if os.path.exists(inPath_1):
		pass
	else:
		os.makedirs(inPath_1)

# 创建栅格
def ArrayToTif(valueArray_1, extent_1, resolution_1, noData_1, outPtah_1):
	lonCount_1 = int((extent_1[3] - extent_1[1]) / resolution_1) + 1
	latCount_1 = int((extent_1[0] - extent_1[2]) / resolution_1) + 1
	driver_1 = gdal.GetDriverByName('GTiff')  #创建.tif文件
	if os.path.exists(outPtah_1):
		pass
	else:
		outTif_1 = driver_1.Create(outPtah_1, lonCount_1, latCount_1, 1, gdal.GDT_Float32) 
        # 设置影像的显示范围，-Lat_Res一定要是-的
		geotransform_1 = (extent_1[1], resolution_1, 0, extent_1[0], 0, -resolution_1)
		outTif_1.SetGeoTransform(geotransform_1)
        #获取地理坐标系统信息，用于选取需要的地理坐标系统
		spatialRef_1 = osr.SpatialReference()
		spatialRef_1.ImportFromEPSG(4326) # 定义输出的坐标系为'WGS 84'，AUTHORITY['EPSG','4326']
		outTif_1.SetProjection(spatialRef_1.ExportToWkt()) # 给新建图层赋予投影信息
        #数据写出
		outTif_1.GetRasterBand(1).WriteArray(valueArray_1) # 将数据写入内存，此时没有写入硬盘
		outTif_1.GetRasterBand(1).SetNoDataValue(noData_1)
		outTif_1.FlushCache() # 将数据写入硬盘
		outTif_1 = None # 注意必须关闭tif文件

# 将单个nc文件转换为多个tif文件
def NcToTif(inPath_1, outPtah_1, variable_1, noData_1, GHWR):
	nc_1 = nc.Dataset(inPath_1)
	lonArray_1 = nc_1.variables['lon'][:]
	latArray_1 = nc_1.variables['lat'][:]
	valueArray_1 = np.asarray(nc_1.variables[variable_1])  #将数据读取为数组
	lonMin_1, latMax_1, lonMax_1, latMin_1 = [lonArray_1.min(), latArray_1.max(), lonArray_1.max(), latArray_1.min()]  #影像的左上角和右下角坐标
	#分辨率计算
	lonCount_1 = len(lonArray_1)
	latCount_1 = len(latArray_1) 
	lonResolution_1 = (lonMax_1 - lonMin_1) / (float(lonCount_1) - 1)
	latResolution_1 = (latMax_1 - latMin_1) / (float(latCount_1) - 1)
	for i in range(len(valueArray_1[:])):
		driver_1 = gdal.GetDriverByName('GTiff')  #创建.tif文件
		# 一个数据变量和多个数据变量
		# outPtah_2 = os.path.join(outPtah_1, GHWR[0] + '_' + GHWR[1] + '_' + GHWR[2], str(i + 1) + '.tif')	# heat wave threshold
		# outPtah_2 = os.path.join(outPtah_1, GHWR[0] + '_' + GHWR[1] + '_' + GHWR[2] + '_' + GHWR[3], inPath_1.split('\\')[-1].split('.')[0][-4:], inPath_1.split('\\')[-1].split('.')[0] + '_' + str(i + 1) + '.tif')	# heat wave record
		# outPtah_2 = os.path.join(outPtah_1, inPath_1.split('\\')[-1].split('.')[0] + '_' + str(i + 1979) + '.tif') # heat wave summary
		outPtah_2 = os.path.join(outPtah_1, inPath_1.split('\\')[-1].split('.')[0], inPath_1.split('\\')[-1].split('.')[1], inPath_1.split('\\')[-1].split('.')[1] + '_' + str(i + 1) + '.tif')	# climate tmax / climate precip
		if os.path.exists(outPtah_2):
			pass
		else:
			outTif_1 = driver_1.Create(outPtah_2, lonCount_1, latCount_1, 1, gdal.GDT_Float32) 
	        # 设置影像的显示范围，-Lat_Res一定要是-的
			geotransform_1 = (lonMin_1, lonResolution_1, 0, latMax_1, 0, -latResolution_1)
			outTif_1.SetGeoTransform(geotransform_1)
	        #获取地理坐标系统信息，用于选取需要的地理坐标系统
			spatialRef_1 = osr.SpatialReference()
			spatialRef_1.ImportFromEPSG(4326) # 定义输出的坐标系为'WGS 84'，AUTHORITY['EPSG','4326']
			outTif_1.SetProjection(spatialRef_1.ExportToWkt()) # 给新建图层赋予投影信息
	        #数据写出
			outTif_1.GetRasterBand(1).WriteArray(valueArray_1[i]) # 将数据写入内存，此时没有写入硬盘
			outTif_1.GetRasterBand(1).SetNoDataValue(noData_1)
			outTif_1.FlushCache() # 将数据写入硬盘
			outTif_1 = None # 注意必须关闭tif文件

# 计算极端气候阈值
def CalExtremeThreshold(inPath_1, inPath_2, yearRange_1, variable_1, outPtah_1):
	dataFrame_1 = pandas.read_csv(inPath_1)	# 读取站点表格
	dataFrame_1.sort_values('stationID', ascending = 1, inplace = True)	# 将站点表格按站点ID排序
	stationIDList_1 = dataFrame_1['stationID'].tolist()
	for date_1 in range(1, 366):
		date_2 = datetime.datetime(2015, 1, 1) + datetime.timedelta(date_1 - 1)
		dayValueArray_1 = np.full((yearRange_1[1] - yearRange_1[0] + 1, len(stationIDList_1)), np.nan)
		index_1 = 0
		for dirPath, dirname, filenames in os.walk(inPath_2):
			for filename in filenames:
				year_1 = filename.split('.')[0][-6: -2]
				month_1 = filename.split('.')[0][-2: ]
				if variable_1 in filename and int(year_1) >= yearRange_1[0] and int(year_1) <= yearRange_1[1] and month_1 == str(date_2.month).zfill(2):
					print(date_1, year_1, month_1, str(date_2.month).zfill(2), filename)
					inPath_3 = os.path.join(dirPath, filename)
					dataFrame_2 = pandas.read_csv(inPath_3)
					dataFrame_3 = dataFrame_2[dataFrame_2['6'] == date_2.day]
					stationIDList_2 = dataFrame_3['0'].tolist()
					for station_1 in stationIDList_1:
						if int(station_1) not in stationIDList_2:
							# print(date_1, year_1, month_1, filename, station_1, stationIDList_1[stationIDList_1.index(station_1) - 1])
							dataFrame_4 = dataFrame_3[dataFrame_3['0'] == stationIDList_1[stationIDList_1.index(station_1) - 1]]
							dataFrame_4.replace(stationIDList_1[stationIDList_1.index(station_1) - 1], stationIDList_1[stationIDList_1.index(station_1)], inplace = True)
							dataFrame_3 = pandas.concat([dataFrame_3, dataFrame_4])
							dataFrame_3.sort_values('0', ascending = 1, inplace = True)
					dayValueArray_1[index_1, :] = dataFrame_3['7'].to_numpy()
					# print(date_1, index_1, filename)
					index_1 = index_1 + 1
		dataFrame_4 = pandas.DataFrame(columns = ['stationID', 'temp_85', 'temp_10'])
		for index_2 in range(len(stationIDList_1)):
			stationValueArray_1 = dayValueArray_1[:, index_2]
			dataFrame_4 = dataFrame_4.append([{'stationID': stationIDList_1[index_2], 'temp_85': np.percentile(stationValueArray_1, 85), 'temp_10': np.percentile(stationValueArray_1, 10)}], ignore_index = True)
		outPtah_2 = os.path.join(outPtah_1, str(date_1) + '.csv')
		dataFrame_4.to_csv(outPtah_2, index = False)
		print(str(date_1) + ' is done!')

# 计算极端气候事件和物候对应表（基于站点数据，以月循环）
def CalExtremeStation(extent_1, dateRange_1, yearRange_1, inPath_1, inPath_2, inPath_3, inPath_4, outPtah_1, item_1):
	dataFrame_1 = pandas.DataFrame(columns = ['range', 'start', 'end', 'stationID', 'latitude', 'longitude', 'altitude', 'landcover', 'temp_mean', 'temp_sum', 'temp_10', 'temp_85', 'prec_sum', 'phot_mean', 'phen'])	# 创建最终存储表格
	dataFrame_2 = pandas.read_csv(inPath_1)	# 读取站点表格
	dataFrame_2.sort_values('stationID', ascending = 1, inplace = True)	# 将站点表格按站点ID排序
	dataFrame_3 = dataFrame_2[(dataFrame_2['latitude'] >= extent_1[0] * 100) & (dataFrame_2['latitude'] <= extent_1[2] * 100) \
	& (dataFrame_2['longitude'] >= extent_1[1] * 100) & (dataFrame_2['longitude'] <= extent_1[3] * 100)]	# 选取在研究区范围内的记录
	stationIDList_1 = dataFrame_3['stationID'].tolist()	# 站点ID列表
	latitudeList_1 = dataFrame_3['latitude_real'].tolist()	# 站点纬度列表
	longitudeList_1 = dataFrame_3['longitude_real'].tolist()	# 站点经度列表
	altitudeList_1 = dataFrame_3['altitude'].tolist()	# 站点海拔列表
	landcoverList_1 = dataFrame_3['landcover'].tolist()	# 站点土地覆被列表
	altitudeList_2 = []
	for altitude_1 in altitudeList_1:
		grade_1 = int(altitude_1 / 300) + 1
		altitudeList_2.append(grade_1)
	stationCount_1 = len(stationIDList_1)	# 站点个数
	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
		dataFrame_4 = pandas.DataFrame(columns = ['range', 'start', 'end', 'stationID', 'latitude', 'longitude', 'altitude', 'landcover', 'temp_mean', 'temp_sum', 'temp_10', 'temp_85', 'prec_sum', 'phot_mean', 'phen'])	# 创建最终存储表格
		inPath_5 = os.path.join(inPath_2, str(year_1) + '_0_Start of Season 1.tif')	# 物候数据路径
		phenRaster_1 = gdal.Open(inPath_5)	# 物候栅格
		phenArray_1 = phenRaster_1.ReadAsArray()	# 物候数组
		phenArray_2 = np.full(stationCount_1, np.nan) # 研究区站点的物候数组
		for index_1 in range(stationCount_1):
			row_1 = int((90 - latitudeList_1[index_1]) / 0.05)	# 纬度对应的行号
			column_1 = 3600 + int(longitudeList_1[index_1] / 0.05)	# 经度对应的列号
			phenArray_2[index_1] = phenArray_1[row_1, column_1]	# 将物候数据存入物候数组
		range_1 = 1	# 时间段标记
		for startDate_1 in range(dateRange_1[0], dateRange_1[1] + 1, 6):
			for endDate_1 in range(startDate_1 + 6, dateRange_1[1] + 1, 6):
				print(year_1, startDate_1, endDate_1)
				dateCount_1 = endDate_1 - startDate_1 + 1
				if startDate_1 < 0 and endDate_1 > 0:
					dateCount_1 = dateCount_1 - 1
				tempArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点气温
				tempArray_2 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内低于极端气温阈值的各站点气温
				tempArray_3 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内超过极端气温阈值的各站点气温
				precArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点降水
				photArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点日照
				meanTempArray_1 = np.full(stationCount_1, np.nan)	# 该时间段内各站点的平均气温
				sumTempArray_1 = np.full(stationCount_1, np.nan)	# 该时间段内各站点的积温
				sumTempArray_2 = np.full(stationCount_1, np.nan)	# 该时间段内各站点低于极端气温阈值的积温
				sumTempArray_3 = np.full(stationCount_1, np.nan)	# 该时间段内各站点超过极端气温阈值的积温
				sumPrecArray_1 = np.full(stationCount_1, np.nan)	# 该时间段内各站点的总降水
				meanPhotArray_1 = np.full(stationCount_1, np.nan)	# 该时间段内各站点的平均日照
				index_2 = 0	# 日期标记
				monthList_1 = []	# 月份列表
				for date_1 in range(startDate_1, endDate_1 + 1):
					if date_1 != 0:
						if date_1 < 0:
							date_2 = datetime.datetime(year_1 - 1, 1, 1) + datetime.timedelta(366 + date_1)
							if date_2.month not in monthList_1:
								monthList_1.append(date_2.month)
						elif date_1 > 0:
							date_2 = datetime.datetime(year_1, 1, 1) + datetime.timedelta(date_1 - 1)
							if date_2.month not in monthList_1:
								monthList_1.append(date_2.month)
				for month_1 in monthList_1:
					if month_1 == 12:
						filename_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001-' + str(year_1 - 1) + '12.csv'	# 气温数据文件名
						filename_2 = 'SURF_CLI_CHN_MUL_DAY-PRE-13011-' + str(year_1 - 1) + '12.csv'	# 降水数据文件名
						filename_3 = 'SURF_CLI_CHN_MUL_DAY-SSD-14032-' + str(year_1 - 1) + '12.csv' # 日照数据文件名
					else:
						filename_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001-' + str(year_1) + str(month_1).zfill(2) + '.csv'	# 气温数据文件名
						filename_2 = 'SURF_CLI_CHN_MUL_DAY-PRE-13011-' + str(year_1) + str(month_1).zfill(2) + '.csv'	# 降水数据文件名
						filename_3 = 'SURF_CLI_CHN_MUL_DAY-SSD-14032-' + str(year_1) + str(month_1).zfill(2) + '.csv'	# 日照数据文件名
					inPath_6 = os.path.join(inPath_3, filename_1)	# 气温数据文件路径
					inPath_7 = os.path.join(inPath_3, filename_2)	# 降水数据文件路径
					inPath_8 = os.path.join(inPath_3, filename_3)	# 日照数据文件路径
					dataFrame_5 = pandas.read_csv(inPath_6)	# 读取气温数据
					dataFrame_6 = pandas.read_csv(inPath_7)	# 读取降水数据
					dataFrame_7 = pandas.read_csv(inPath_8)	# 读取日照数据
					dataFrame_5.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'tem_mean', 'tem_maximum', 'tem_minimum', 'tem_mean_QC', 'tem_maximum_QC', 'tem_minimum_QC']	# 修改气温数据列名
					dataFrame_6.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'pre_20-8', 'pre_8-20', 'pre_20-20', 'pre_20-8_QC', 'pre_8-20_QC', 'pre_20-20_QC']	# 修改降水数据列名	
					dataFrame_7.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'ssd', 'ssd_QC']	# 修改日照数据列名
					for date_3 in range(startDate_1, endDate_1 + 1):
						if date_3 != 0:
							if date_3 < 0:
								date_4 = datetime.datetime(year_1 - 1, 1, 1) + datetime.timedelta(366 + date_3)
							elif date_3 > 0:
								date_4 = datetime.datetime(year_1, 1, 1) + datetime.timedelta(date_3 - 1)
							if date_4.month == month_1:
								dataFrame_8 = dataFrame_5[(dataFrame_5['day'] == date_4.day) & (dataFrame_5['stationID'].isin(stationIDList_1))].copy()	# 选取该日期且在研究区范围内的记录
								dataFrame_9 = dataFrame_6[(dataFrame_6['day'] == date_4.day) & (dataFrame_6['stationID'].isin(stationIDList_1))].copy()	# 选取该日期且在研究区范围内的记录
								dataFrame_10 = dataFrame_7[(dataFrame_7['day'] == date_4.day) & (dataFrame_7['stationID'].isin(stationIDList_1))].copy()	# 选取该日期且在研究区范围内的记录
								dataFrame_8.loc[(dataFrame_8['tem_mean'] < -600) | (dataFrame_8['tem_mean'] > 600), 'tem_mean'] = np.nan
								dataFrame_9.loc[dataFrame_9['pre_20-20'] > 30000, 'pre_20-20'] = np.nan
								dataFrame_10.loc[dataFrame_10['ssd'] > 240, 'ssd'] = np.nan
								# dataFrame_8.sort_values('stationID', ascending = 1, inplace = True)	# 按站点ID排序
								# dataFrame_9.sort_values('stationID', ascending = 1, inplace = True)	# 按站点ID排序
								tempArray_1[index_2, :] = dataFrame_8['tem_mean'].to_numpy()	# 将气温数据写入数组
								tempArray_2[index_2, :] = dataFrame_8['tem_mean'].to_numpy()	# 将气温数据写入数组
								tempArray_3[index_2, :] = dataFrame_8['tem_mean'].to_numpy()	# 将气温数据写入数组
								precArray_1[index_2, :] = dataFrame_9['pre_20-20'].to_numpy()	# 将降水数据写入数组
								photArray_1[index_2, :] = dataFrame_10['ssd'].to_numpy()	# 将降水数据写入数组
								if date_3 < 0:
									inPath_9 = os.path.join(inPath_4, str(366 + date_3) + '.csv')	# 该日极端气候阈值文件路径
								elif date_3 > 0:
									inPath_9 = os.path.join(inPath_4, str(date_3) + '.csv')	# 该日极端气候阈值文件路径
								dataFrame_11 = pandas.read_csv(inPath_9)	# 该日极端气候阈值文件
								for index_3 in range(stationCount_1):	# 遍历该日每个站点
									station_1 = dataFrame_8['stationID'].tolist()[index_3]	# 获取站点ID
									threshold_1 = dataFrame_11[dataFrame_11['stationID'] == station_1]['temp_10'].to_numpy()[0]	# 获取该站点的阈值
									threshold_2 = dataFrame_11[dataFrame_11['stationID'] == station_1]['temp_85'].to_numpy()[0]	# 获取该站点的阈值
									if tempArray_2[index_2, index_3] > threshold_1:
										tempArray_2[index_2, index_3] = 0
									if tempArray_3[index_2, index_3] < threshold_2:
										tempArray_3[index_2, index_3] = 0
								index_2 = index_2 + 1
				for index_4 in range(stationCount_1):
					meanTempArray_1[index_4] = np.nanmean(tempArray_1[:, index_4])	# 该时间段内各站点的平均气温
					sumTempArray_1[index_4] = np.nansum(tempArray_1[:, index_4])	# 该时间段内各站点的积温
					sumTempArray_2[index_4] = np.nansum(tempArray_2[:, index_4])	# 该时间段内各站点的低于极端气候阈值的积温
					sumTempArray_3[index_4] = np.nansum(tempArray_3[:, index_4])	# 该时间段内各站点的超过极端气候阈值的积温
					sumPrecArray_1[index_4] = np.nansum(precArray_1[:, index_4])	# 该时间段内各站点的平均降水
					meanPhotArray_1[index_4] = np.nanmean(photArray_1[:, index_4])	# 该时间段内各站点的平均日照
				rangeArray_1 = np.full(stationCount_1, range_1)	# 时间段标记数组
				startDateArray_1 = np.full(stationCount_1, startDate_1)	# 开始日期标记数组
				endDateArray_1 = np.full(stationCount_1, endDate_1)	# 结束日期标记数组
				dataFrame_12 = pandas.DataFrame({'range': rangeArray_1, 'start': startDateArray_1, 'end': endDateArray_1, 'stationID': stationIDList_1, \
					'latitude': latitudeList_1, 'longitude': longitudeList_1, 'altitude': altitudeList_2, 'landcover': landcoverList_1, 'temp_mean': meanTempArray_1, \
					'temp_sum': sumTempArray_1, 'temp_10': sumTempArray_2, 'temp_85': sumTempArray_3, 'prec_sum': sumPrecArray_1, 'phot_mean': meanPhotArray_1, 'phen': phenArray_2})	# 将记录写入表格
				dataFrame_4 = pandas.concat([dataFrame_4, dataFrame_12])
				dataFrame_1 = pandas.concat([dataFrame_1, dataFrame_12])	# 表格拼接
				range_1 = range_1 + 1
		outPtah_2 = os.path.join(outPtah_1, 'items_' + item_1 + '_' + str(year_1) + '.csv')
		dataFrame_4.to_csv(outPtah_2, index = False)	# 保存文件
	outPtah_3 = os.path.join(outPtah_1, 'items_' + item_1 + '.csv')
	dataFrame_1.to_csv(outPtah_3, index = False)	# 保存文件

# 计算各变量和物候的统计值
def CalStatistic_1(inPath_1, variableList_1, columnList_1, outPtah_1, item_1):
	dataFrame_1 = pandas.DataFrame(columns = columnList_1)
	dataFrame_2 = pandas.read_csv(inPath_1)
	for variable_1 in variableList_1:
		variableArray_1 = dataFrame_2[variable_1].to_numpy()
		# variableArray_2 = preprocessing.StandardScaler().fit_transform(variableArray_1.reshape(-1, 1))
		# dataFrame_2[variable_1 + '_nom'] = variableArray_2.reshape(1, -1)[0]
		# variableArray_1 = variableArray_1[~np.isnan(variableArray_1)]
		for index in range(len(variableArray_1)):
			if np.isnan(variableArray_1[index]):
				print(index)
		variableArray_2 = preprocessing.Normalizer().fit_transform(variableArray_1.reshape(1, -1))
		dataFrame_2[variable_1 + '_nom'] = variableArray_2[0]
	phenArray_1 = dataFrame_2['phen'].to_numpy()
	# phenArray_2 = preprocessing.StandardScaler().fit_transform(phenArray_1.reshape(-1, 1))
	# dataFrame_2['phen_nom'] = phenArray_2.reshape(1, -1)[0]
	phenArray_2 = preprocessing.Normalizer().fit_transform(phenArray_1.reshape(1, -1))
	dataFrame_2['phen_nom'] = phenArray_2[0]
	rangeArray_1 = dataFrame_2['range'].unique()
	for range_1 in rangeArray_1:
		dataFrame_3 = dataFrame_2[(dataFrame_2['range'] == range_1)]
		landcoverArray_1 = dataFrame_3['landcover'].unique()
		for landcover_1 in landcoverArray_1:
			dataFrame_4 = dataFrame_3[(dataFrame_3['landcover'] == landcover_1)]
			startDate_1 = dataFrame_4['start'].unique()[0]
			endDate_1 = dataFrame_4['end'].unique()[0]
			phenArray_3 = dataFrame_4['phen_nom'].to_numpy()
			valueList_1 = [range_1, startDate_1, endDate_1, landcover_1]
			for variable_1 in variableList_1:
				variableArray_3 = dataFrame_4[variable_1 + '_nom'].to_numpy()
				slopeVariable_1, interceptVariable_1, rValueVariable_1, pValueVariable_1, stdErrVariable_1 = scipy.stats.linregress(variableArray_3.tolist(), phenArray_3.tolist())
				valueList_1.append(slopeVariable_1)
				valueList_1.append(rValueVariable_1 * rValueVariable_1)
				valueList_1.append(pValueVariable_1)
			dataFrame_1.loc[len(dataFrame_1)] = valueList_1
	# outPtah_2 = os.path.join(outPtah_1, 'result_' + item_1 + '_Standard.csv')
	outPtah_2 = os.path.join(outPtah_1, 'result_' + item_1 + '_Normalize.csv')
	dataFrame_1.to_csv(outPtah_2, index = False)

	dataFrame_5 = pandas.DataFrame(columns = ['range', 'start', 'end', 'landcover', 'type', 'value'])
	rangeArray_2 = dataFrame_1['range'].to_numpy()
	startDateArray_1 = dataFrame_1['start'].to_numpy()
	endDateArray_1 = dataFrame_1['end'].to_numpy()
	landcoverArray_2 = dataFrame_1['landcover'].to_numpy()
	for variable_2 in columnList_1[4: ]:
		typeArray_1 = np.full(len(dataFrame_1), variable_2)
		variableArray_4 = dataFrame_1[variable_2].to_numpy()
		dataFrame_6 = pandas.DataFrame({'range': rangeArray_2, 'start': startDateArray_1, 'end': endDateArray_1, 'landcover': landcoverArray_2, 'type': typeArray_1, 'value': variableArray_4})
		dataFrame_5 = pandas.concat([dataFrame_5, dataFrame_6])
	outPtah_3 = os.path.join(outPtah_1, 'result_' + item_1 + '_Normalize_Long.csv')
	dataFrame_5.to_csv(outPtah_3, index = False)


# 计算各变量和物候的统计值
def CalStatistic_2(inPath_1, variableList_1, columnList_1, outPtah_1, item_1):
	dataFrame_1 = pandas.DataFrame(columns = columnList_1)
	dataFrame_2 = pandas.read_csv(inPath_1)
	rangeArray_1 = dataFrame_2['range'].unique()
	for range_1 in rangeArray_1:
		dataFrame_3 = dataFrame_2[(dataFrame_2['range'] == range_1)].copy()
		landcoverArray_1 = dataFrame_3['landcover'].unique()
		for landcover_1 in landcoverArray_1:
			dataFrame_4 = dataFrame_3[(dataFrame_3['landcover'] == landcover_1)].copy()
			X_1 = dataFrame_4[variableList_1]
			y_1 = dataFrame_4[['phen']]
			XTrain_1, XTest_1, yTrain_1, yTest_1 = train_test_split(X_1, y_1, test_size = 0.3, random_state = 0) 
			pls_1 = PLSRegression(n_components = 3)
			pls_1.fit(scale(XTrain_1), yTrain_1)
			coefArray_1 = pls_1.coef_.reshape(1, -1)[0]
			xScores_1 = pls_1.x_scores_
			xWeights_1 = pls_1.x_weights_
			yLoadings_1 = pls_1.y_loadings_
			rows_1, columns_1 = xWeights_1.shape
			vipArray_1 = np.zeros((rows_1,))
			all_1 = np.diag(xScores_1.T @ xScores_1 @ yLoadings_1.T @ yLoadings_1).reshape(columns_1, -1)
			alls_1 = np.sum(all_1)
			for row_1 in range(rows_1):
				weight_1 = np.array([(xWeights_1[row_1, column_1] / np.linalg.norm(xWeights_1[:, column_1])) ** 2 for column_1 in range(columns_1)])
				vipArray_1[row_1] = np.sqrt(rows_1 * (all_1.T @ weight_1) / alls_1)
			startDate_1 = dataFrame_4['start'].unique()[0]
			endDate_1 = dataFrame_4['end'].unique()[0]
			valueList_1 = [range_1, startDate_1, endDate_1, landcover_1, coefArray_1[0], vipArray_1[0], coefArray_1[1], vipArray_1[1], coefArray_1[2], vipArray_1[2], coefArray_1[3], \
			vipArray_1[3], coefArray_1[4], vipArray_1[4], coefArray_1[5],  vipArray_1[5]]
			dataFrame_1.loc[len(dataFrame_1)] = valueList_1
		print(str(range_1) + ' is done!')
	outPtah_2 = os.path.join(outPtah_1, 'result_' + item_1 + '.csv')
	dataFrame_1.to_csv(outPtah_2, index = False)
	dataFrame_5 = pandas.DataFrame(columns = ['range', 'start', 'end', 'landcover', 'type', 'value'])
	rangeArray_2 = dataFrame_1['range'].to_numpy()
	startDateArray_1 = dataFrame_1['start'].to_numpy()
	endDateArray_1 = dataFrame_1['end'].to_numpy()
	landcoverArray_2 = dataFrame_1['landcover'].to_numpy()
	for variable_2 in columnList_1[4: ]:
		typeArray_1 = np.full(len(dataFrame_1), variable_2)
		variableArray_4 = dataFrame_1[variable_2].to_numpy()
		dataFrame_6 = pandas.DataFrame({'range': rangeArray_2, 'start': startDateArray_1, 'end': endDateArray_1, 'landcover': landcoverArray_2, 'type': typeArray_1, 'value': variableArray_4})
		dataFrame_5 = pandas.concat([dataFrame_5, dataFrame_6])
	outPtah_3 = os.path.join(outPtah_1, 'result_' + item_1 + '_long.csv')
	dataFrame_5.to_csv(outPtah_3, index = False)

def main():
	# # 计算极端气候事件和物候对应表（基于栅格数据）
	# extent_1 = [36, 105, 31, 114]
	# dateRange_1 = [-30, 30]
	# yearRange_1 = [2015, 2016]
	# inPath_1 = r'H:\2_master_paper\data\1_climate\2_tif'
	# inPath_2 = r'H:\2_master_paper\data\3_phenology\0_Start of Season 1_resample'
	# variable_1 = 'tmax'
	# outPtah_1 = r'H:\2_master_paper\data\test\test_3.csv'
	# CalExtremeTable2(extent_1, dateRange_1, yearRange_1, inPath_1, inPath_2, variable_1, outPtah_1)

	# # 计算极端气候阈值
	# inPath_1 = r'H:\2_master_paper\data\5_observation\0_base\station_lc_3.csv'
	# inPath_2 = r'H:\2_master_paper\data\5_observation\3_select'
	# yearRange_1 = [1987, 2016]
	# outPtah_1 = r'H:\2_master_paper\data\5_observation\5_threshold\10_85'
	# variable_1 = 'TEM'
	# CalExtremeThreshold(inPath_1, inPath_2, yearRange_1, variable_1, outPtah_1)

	# 计算极端气候事件和物候对应表（基于站点数据，以月循环）
	extent_1 = [32, 105, 35, 114]	# 秦岭
	# extent_1 = [47, 117, 54, 128]	# 东北
	dateRange_1 = [-32, 151]
	yearRange_1 = [1987, 2016]
	item_1 = 'Qinling_' + str(yearRange_1[0]) + '-' + str(yearRange_1[1]) + '_' + str(dateRange_1[0]) + '-' + str(dateRange_1[1])
	inPath_1 = r'H:\2_master_paper\data\5_observation\0_base\station_lc_3.csv'
	inPath_2 = r'H:\2_master_paper\data\3_phenology\0_Start of Season 1'
	inPath_3 = r'H:\2_master_paper\data\5_observation\3_select'
	inPath_4 = r'H:\2_master_paper\data\5_observation\5_threshold\10_85'
	outPtah_1 = r'H:\2_master_paper\data\6_result\秦岭\1987-2016_-32-151'
	# CalExtremeStation(extent_1, dateRange_1, yearRange_1, inPath_1, inPath_2, inPath_3, inPath_4, outPtah_1, item_1)

	# # 将各年份的表格合并起来
	# inPath_1 = r'H:\2_master_paper\data\6_result\秦岭\1987-2016_-32-151\items'
	# dataFrame_1 = pandas.DataFrame(columns = ['range', 'start', 'end', 'stationID', 'latitude', 'longitude', 'altitude', 'landcover', 'temp_mean', 'temp_sum', 'temp_10', 'temp_85', 'prec_sum', 'phot_mean', 'phen'])
	# for year_1 in range(1987, 2017):
	# 	inPath_2 = os.path.join(inPath_1, 'items_Qinling_1987-2016_-32-151_' + str(year_1) + '.csv')
	# 	dataFrame_2 = pandas.read_csv(inPath_2)
	# 	dataFrame_1 = pandas.concat([dataFrame_1, dataFrame_2])
	# 	print(str(year_1) + ' is done!')
	# outPtah_1 = r'H:\2_master_paper\data\6_result\秦岭\1987-2016_-32-151'
	# outPtah_2 = os.path.join(outPtah_1, 'items_Qinling_1987-2016_-32-151' + '.csv')
	# dataFrame_1.to_csv(outPtah_2, index = False)

	# 计算各变量和物候的统计值
	inPath_5 = os.path.join(outPtah_1, 'items_' + item_1 + '.csv')
	variableList_1 = ['temp_mean', 'temp_sum', 'temp_10', 'temp_85', 'prec_sum', 'phot_mean']
	columnList_1 = ['range', 'start', 'end', 'landcover']
	for variable_1 in variableList_1:
		columnList_1.append(variable_1 + 'Coef')
		columnList_1.append(variable_1 + 'Vip')
	CalStatistic_2(inPath_5, variableList_1, columnList_1, outPtah_1, item_1)

main()