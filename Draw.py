import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from osgeo import gdal, ogr
import numpy as np
import pandas
import os
import math
import scipy.stats
import matplotlib.font_manager as fm
import datetime
from plotnine import *
from sklearn import preprocessing
import pylab
from datetime import datetime, date, timedelta
# props = fm.FontProperties(fname = r'C:\Windows\Fonts\simhei.ttf')
plt.rc('font', family = 'Times New Roman')
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

# 绘制站点数据线盒图（未标准化）
def DrawStationBox_1(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, meanList_1, stdList_1, extremeList_1, tags_1, color_1, xlabel_1, outPtah_1, textLocationlist_1, figureList_1):
	dataFrame_1 = pandas.read_csv(inPath_1)	# 读取站点表格
	dataFrame_1.sort_values('stationID', ascending = 1, inplace = True)	# 将站点表格按站点ID排序
	dataFrame_2 = dataFrame_1[(dataFrame_1['latitude'] >= extent_1[0] * 100) & (dataFrame_1['latitude'] <= extent_1[2] * 100) \
	& (dataFrame_1['longitude'] >= extent_1[1] * 100) & (dataFrame_1['longitude'] <= extent_1[3] * 100)]	# 选取在研究区范围内的记录
	stationIDList_1 = dataFrame_2['stationID'].tolist()	# 站点ID列表
	stationCount_1 = len(stationIDList_1)	# 站点ID列表长度
	dateCount_1 = dateRange_1[1] - dateRange_1[0] + 1	# 日数
	if dateRange_1[0] < 0 and dateRange_1[1] > 0:	# 排除day = 0
		dateCount_1 = dateCount_1 - 1
	meanTempList_1 = [] 	# 存储各站点平均气温
	sumTempList_1 = []	# 存储各站点积温
	sumTempList_2 = []	# 存储各站点低于极端温度阈值的积温
	sumTempList_3 = []	# 存储各站点高于极端温度阈值的积温
	sumPrecList_1 = []	# 存储各站点总降水量
	meanPhotList_1 = []
	chillingList_1 = []
	forcingList_1 = []
	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
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
		meanPhotArray_1 = np.full(stationCount_1, np.nan)
		sumChillingArray_1 = np.full(stationCount_1, 0)
		sumForcingArray_1 = np.full(stationCount_1, 0)
		index_1 = 0
		for date_1 in range(dateRange_1[0], dateRange_1[1] + 1):
			if date_1 != 0:
				if date_1 < 0:
					date_2 = datetime.datetime(year_1 - 1, 1, 1) + datetime.timedelta(366 + date_1)
				elif date_1 > 0:
					date_2 = datetime.datetime(year_1, 1, 1) + datetime.timedelta(date_1 - 1)
				print(date_2)
				filename_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 气温数据文件名
				filename_2 = 'SURF_CLI_CHN_MUL_DAY-PRE-13011-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 降水数据文件名
				filename_3 = 'SURF_CLI_CHN_MUL_DAY-SSD-14032-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 日照数据文件名
				inPath_4 = os.path.join(inPath_2, filename_1)
				inPath_5 = os.path.join(inPath_2, filename_2)
				inPath_6 = os.path.join(inPath_2, filename_3)
				dataFrame_3 = pandas.read_csv(inPath_4)
				dataFrame_4 = pandas.read_csv(inPath_5)
				dataFrame_5 = pandas.read_csv(inPath_6)
				dataFrame_3.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'tem_mean', 'tem_maximum', 'tem_minimum', 'tem_mean_QC', 'tem_maximum_QC', 'tem_minimum_QC']	# 修改气温数据列名
				dataFrame_4.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'pre_20-8', 'pre_8-20', 'pre_20-20', 'pre_20-8_QC', 'pre_8-20_QC', 'pre_20-20_QC']	# 修改降水数据列名	
				dataFrame_5.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'ssd', 'ssd_QC']	# 修改日照数据列名		
				dataFrame_6 = dataFrame_3[(dataFrame_3['day'] == date_2.day) & (dataFrame_3['stationID'].isin(stationIDList_1))]
				dataFrame_7 = dataFrame_4[(dataFrame_4['day'] == date_2.day) & (dataFrame_4['stationID'].isin(stationIDList_1))]
				dataFrame_8 = dataFrame_5[(dataFrame_5['day'] == date_2.day) & (dataFrame_5['stationID'].isin(stationIDList_1))]
				tempArray_1[index_1, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				tempArray_2[index_1, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				tempArray_3[index_1, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				precArray_1[index_1, :] = dataFrame_7['pre_20-20'].to_numpy()	# 将降水数据写入数组		
				photArray_1[index_1, :] = dataFrame_8['ssd'].to_numpy()	# 将降水数据写入数组
				if date_1 < 0:
					inPath_7 = os.path.join(inPath_3, str(366 + date_1) + '.csv')	# 该日极端气候阈值文件路径
				elif date_1 > 0:
					inPath_7 = os.path.join(inPath_3, str(date_1) + '.csv')	# 该日极端气候阈值文件路径
				dataFrame_9 = pandas.read_csv(inPath_7)	# 该日极端气候阈值文件
				for index_2 in range(stationCount_1):	# 遍历该日每个站点
					station_1 = dataFrame_6['stationID'].tolist()[index_2]	# 获取站点ID
					threshold_1 = dataFrame_9[dataFrame_9['stationID'] == station_1]['temp_10'].to_numpy()[0]	# 获取该站点的阈值
					threshold_2 = dataFrame_9[dataFrame_9['stationID'] == station_1]['temp_85'].to_numpy()[0]	# 获取该站点的阈值
					if tempArray_2[index_1, index_2] > threshold_1:
						tempArray_2[index_1, index_2] = 0
					if tempArray_3[index_1, index_2] < threshold_2:
						tempArray_3[index_1, index_2] = 0
					if tempArray_2[index_1, index_2] <= threshold_1:
						sumChillingArray_1[index_2] = sumChillingArray_1[index_2] + 1
					if tempArray_3[index_1, index_2] >= threshold_2:
						sumForcingArray_1[index_2] = sumForcingArray_1[index_2] + 1
				index_1 = index_1 + 1
		for index_3 in range(stationCount_1):
			tempList_1 = tempArray_1[:, index_3].tolist()
			for temp_1 in tempList_1:
				if temp_1 < -600 or temp_1 > 600:
					print('temp: ' + str(temp_1))
					tempList_1.remove(temp_1)
			tempList_2 = tempArray_2[:, index_3].tolist()
			for temp_2 in tempList_2:
				if temp_2 < -600 or temp_2 > 600:
					print('temp: ' + str(temp_2))
					tempList_2.remove(temp_2)
			tempList_3 = tempArray_3[:, index_3].tolist()
			for temp_3 in tempList_3:
				if temp_3 < -600 or temp_3 > 600:
					print('temp: ' + str(temp_3))
					tempList_3.remove(temp_3)
			precList_1 = precArray_1[:, index_3].tolist()
			for prec_1 in precList_1:
				if prec_1 > 30000:
					print('prec: ' + str(prec_1))
					precList_1.remove(prec_1)
			photList_1 = photArray_1[:, index_3].tolist()
			for phot_1 in photList_1:
				if phot_1 > 240:
					print('phot: ' + str(phot_1))
					photList_1.remove(phot_1)
			meanTempArray_1[index_3] = np.mean(np.array(tempList_1))	# 该时间段内各站点的平均气温
			sumTempArray_1[index_3] = np.sum(np.array(tempList_1))	# 该时间段内各站点的积温
			sumTempArray_2[index_3] = np.sum(np.array(tempList_2))	# 该时间段内各站点的低于极端气候阈值的积温
			sumTempArray_3[index_3] = np.sum(np.array(tempList_3))	# 该时间段内各站点的超过极端气候阈值的积温
			sumPrecArray_1[index_3] = np.sum(np.array(precList_1))	# 该时间段内各站点的平均降水
			meanPhotArray_1[index_3] = np.mean(np.array(photList_1))
		# meanTempArray_2 = (meanTempArray_1 - meanList_1[0]) / stdList_1[0]
		# sumTempArray_4 = (sumTempArray_1 - meanList_1[1]) / stdList_1[1]
		# sumTempArray_5 = (sumTempArray_2 - meanList_1[2]) / stdList_1[2]
		# sumTempArray_6 = (sumTempArray_3 - meanList_1[3]) / stdList_1[3]
		# sumPrecArray_2 = (sumPrecArray_1 - meanList_1[4]) / stdList_1[4]
		meanTempArray_2 = (meanTempArray_1 - meanList_1[0]) / 10
		sumTempArray_4 = (sumTempArray_1 - meanList_1[1]) / 10
		sumTempArray_5 = (sumTempArray_2 - meanList_1[2]) / 10
		sumTempArray_6 = (sumTempArray_3 - meanList_1[3]) / 10
		sumPrecArray_2 = (sumPrecArray_1 - meanList_1[4]) / 1000
		meanPhotArray_2 = (meanPhotArray_1 - meanList_1[5]) / 10
		# meanTempArray_2 = meanTempArray_1 / 10
		# sumTempArray_4 = sumTempArray_1 / 10
		# sumTempArray_5 = sumTempArray_2 / 10
		# sumTempArray_6 = sumTempArray_3 / 10
		# sumPrecArray_2 = sumPrecArray_1 / 1000
		meanTempList_1.append(meanTempArray_2.tolist())
		sumTempList_1.append(sumTempArray_4.tolist())
		sumTempList_2.append(sumTempArray_5.tolist())
		sumTempList_3.append(sumTempArray_6.tolist())
		sumPrecList_1.append(sumPrecArray_2.tolist())
		meanPhotList_1.append(meanPhotArray_2.tolist())
		chillingList_1.append(sumChillingArray_1.tolist())
		forcingList_1.append(sumForcingArray_1.tolist())
	index_4 = 0
	for valueList_1 in [meanTempList_1, sumTempList_1, sumTempList_2, sumTempList_3, sumPrecList_1, meanPhotList_1, chillingList_1, forcingList_1]:
		xList_1 = []
		labels_1 = []
		index_5 = 0
		for year in range(yearRange_1[0], yearRange_1[1] + 1):
			xList_1.append(index_5 + 1)
			labels_1.append(str(year))
			index_5 = index_5 + 1
		plt.figure(figsize = (13, 5))
		plt.subplots_adjust(left = 0.06, bottom = 0.18, right = 0.98, top = 0.9)	

		yList_1 = []
		for valueList_2 in valueList_1:
			yList_1.append(np.median(np.array(valueList_2)))
		plt.scatter(xList_1, yList_1, c = 'black', s = 6, zorder = 3)
		fit_1 = np.polyfit(xList_1, yList_1, 1)
		ld_1 = np.poly1d(fit_1)
		pylab.plot(xList_1, ld_1(xList_1), c = 'black', zorder = 4)
		slopeVariable_1, interceptVariable_1, rValueVariable_1, pValueVariable_1, stdErrVariable_1 = scipy.stats.linregress(xList_1, yList_1)
		function_1 = 'y=' + str(ld_1).replace(' ', '') + ', R=' + str(round(rValueVariable_1, 2))
		if pValueVariable_1 < 0.05:
			function_2 = function_1 + ', P<0.05'
		else:
			function_2 = function_1 + ', P≥0.05'
		plt.text(textLocationlist_1[index_4][0], textLocationlist_1[index_4][1], function_2.replace('\n', ''), fontsize = 16)

		boxplot_1 = plt.boxplot(valueList_1, labels = labels_1, showfliers = False, patch_artist = True, zorder = 2)
		plt.setp(boxplot_1['medians'], color = 'black')
		medianList_1 = []
		for medianLine_1 in boxplot_1['medians']:
			median_1 = medianLine_1.get_ydata()[0]
			medianList_1.append(median_1)
		tag_1 = 0
		if index_4 < 6:
			for patch in boxplot_1['boxes']:
				if medianList_1[tag_1] > extremeList_1[index_4]:
					patch.set(facecolor = color_1)
				elif medianList_1[tag_1] < -extremeList_1[index_4]:
					pass
				else:
					patch.set(facecolor = 'lightgrey')
				tag_1 =  tag_1 + 1
			plt.yticks(list(plt.yticks()[0]) + [extremeList_1[index_4], -extremeList_1[index_4]], fontsize = 16)
			plt.axhline(y = -extremeList_1[index_4], c = 'grey', ls = 'dashed', zorder = 1)
			plt.axhline(y = extremeList_1[index_4], c = 'grey', ls = 'dashed', zorder = 1)
		else:
			plt.yticks(fontsize = 14)
			for patch in boxplot_1['boxes']:
				patch.set(facecolor = 'lightgrey')			
		plt.axhline(y = 0, c = 'grey', ls = 'dashed', zorder = 1)
		plt.title(label = figureList_1[index_4], fontsize = 24)
		plt.xlabel(xlabel_1, fontsize = 18)
		# plt.ylabel(figureList_1[index_4], fontsize = 16)
		plt.xticks(fontsize = 16, rotation = 45)
		outPtah_2 = os.path.join(outPtah_1, str(index_4 + 1) + '_' + figureList_1[index_4] + '.jpg')
		outPtah_3 = os.path.join(outPtah_1, str(index_4 + 1) + '_' + figureList_1[index_4] + '.pdf')
		plt.savefig(outPtah_2, dpi = 330)
		plt.savefig(outPtah_3, dpi = 330)
		# plt.show()
		index_4 = index_4 + 1

# 绘制站点数据线盒图（标准化）
def DrawStationBox_2(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, extremeList_1, tags_1, color_1, xlabel_1, outPtah_1, textLocationlist_1, figureList_1):
	dataFrame_1 = pandas.read_csv(inPath_1)	# 读取站点表格
	dataFrame_1.sort_values('stationID', ascending = 1, inplace = True)	# 将站点表格按站点ID排序
	dataFrame_2 = dataFrame_1[(dataFrame_1['latitude'] >= extent_1[0] * 100) & (dataFrame_1['latitude'] <= extent_1[2] * 100) \
	& (dataFrame_1['longitude'] >= extent_1[1] * 100) & (dataFrame_1['longitude'] <= extent_1[3] * 100)]	# 选取在研究区范围内的记录
	stationIDList_1 = dataFrame_2['stationID'].tolist()	# 站点ID列表
	stationCount_1 = len(stationIDList_1)	# 站点ID列表长度
	yearCount_1 = yearRange_1[1] - yearRange_1[0] + 1	# 年份数
	dateCount_1 = dateRange_1[1] - dateRange_1[0] + 1	# 日数
	if dateRange_1[0] < 0 and dateRange_1[1] > 0:	# 排除day = 0
		dateCount_1 = dateCount_1 - 1
	meanTempArray_1 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点的平均气温
	sumTempArray_1 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点的积温
	sumTempArray_2 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点低于极端气温阈值的积温
	sumTempArray_3 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点超过极端气温阈值的积温
	sumPrecArray_1 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点的总降水
	meanPhotArray_1 = np.full(stationCount_1 * yearCount_1, np.nan)	# 该时间段内各站点的平均日照
	index_1 = 0
	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
		tempArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点气温
		tempArray_2 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内低于极端气温阈值的各站点气温
		tempArray_3 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内超过极端气温阈值的各站点气温
		precArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点降水
		photArray_1 = np.full((dateCount_1, stationCount_1), np.nan)	# 该时间段内各站点日照
		index_2 = 0
		for date_1 in range(dateRange_1[0], dateRange_1[1] + 1):
			if date_1 != 0:
				if date_1 < 0:
					date_2 = datetime.datetime(year_1 - 1, 1, 1) + datetime.timedelta(366 + date_1)
				elif date_1 > 0:
					date_2 = datetime.datetime(year_1, 1, 1) + datetime.timedelta(date_1 - 1)
				filename_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 气温数据文件名
				filename_2 = 'SURF_CLI_CHN_MUL_DAY-PRE-13011-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 降水数据文件名
				filename_3 = 'SURF_CLI_CHN_MUL_DAY-SSD-14032-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 日照数据文件名
				inPath_4 = os.path.join(inPath_2, filename_1)
				inPath_5 = os.path.join(inPath_2, filename_2)
				inPath_6 = os.path.join(inPath_2, filename_3)
				print(date_2)
				dataFrame_3 = pandas.read_csv(inPath_4)
				dataFrame_4 = pandas.read_csv(inPath_5)
				dataFrame_5 = pandas.read_csv(inPath_6)
				dataFrame_3.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'tem_mean', 'tem_maximum', 'tem_minimum', 'tem_mean_QC', 'tem_maximum_QC', 'tem_minimum_QC']	# 修改气温数据列名
				dataFrame_4.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'pre_20-8', 'pre_8-20', 'pre_20-20', 'pre_20-8_QC', 'pre_8-20_QC', 'pre_20-20_QC']	# 修改降水数据列名	
				dataFrame_5.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'ssd', 'ssd_QC']	# 修改日照数据列名		
				dataFrame_6 = dataFrame_3[(dataFrame_3['day'] == date_2.day) & (dataFrame_3['stationID'].isin(stationIDList_1))]
				dataFrame_7 = dataFrame_4[(dataFrame_4['day'] == date_2.day) & (dataFrame_4['stationID'].isin(stationIDList_1))]
				dataFrame_8 = dataFrame_5[(dataFrame_5['day'] == date_2.day) & (dataFrame_5['stationID'].isin(stationIDList_1))]
				tempArray_1[index_2, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				tempArray_2[index_2, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				tempArray_3[index_2, :] = dataFrame_6['tem_mean'].to_numpy()	# 将气温数据写入数组
				precArray_1[index_2, :] = dataFrame_7['pre_20-20'].to_numpy()	# 将降水数据写入数组
				photArray_1[index_2, :] = dataFrame_8['ssd'].to_numpy()	# 将降水数据写入数组
				if date_1 < 0:
					inPath_7 = os.path.join(inPath_3, str(366 + date_1) + '.csv')	# 该日极端气候阈值文件路径
				elif date_1 > 0:
					inPath_7 = os.path.join(inPath_3, str(date_1) + '.csv')	# 该日极端气候阈值文件路径
				dataFrame_9 = pandas.read_csv(inPath_7)	# 该日极端气候阈值文件
				for index_3 in range(stationCount_1):	# 遍历该日每个站点
					station_1 = dataFrame_6['stationID'].tolist()[index_3]	# 获取站点ID
					threshold_1 = dataFrame_9[dataFrame_9['stationID'] == station_1]['temp_10'].to_numpy()[0]	# 获取该站点的阈值
					threshold_2 = dataFrame_9[dataFrame_9['stationID'] == station_1]['temp_85'].to_numpy()[0]	# 获取该站点的阈值
					if tempArray_2[index_2, index_3] > threshold_1:
						tempArray_2[index_2, index_3] = 0
					if tempArray_3[index_2, index_3] < threshold_2:
						tempArray_3[index_2, index_3] = 0
				index_2 = index_2 + 1
		for index_4 in range(stationCount_1):
			tempList_1 = tempArray_1[:, index_4].tolist()
			for temp_1 in tempList_1:
				if temp_1 < -600 or temp_1 > 600:
					print('temp: ' + str(temp_1))
					tempList_1.remove(temp_1)
			tempList_2 = tempArray_2[:, index_4].tolist()
			for temp_2 in tempList_2:
				if temp_2 < -600 or temp_2 > 600:
					print('temp: ' + str(temp_2))
					tempList_2.remove(temp_2)
			tempList_3 = tempArray_3[:, index_4].tolist()
			for temp_3 in tempList_3:
				if temp_3 < -600 or temp_3 > 600:
					print('temp: ' + str(temp_3))
					tempList_3.remove(temp_3)
			precList_1 = precArray_1[:, index_4].tolist()
			for prec_1 in precList_1:
				if prec_1 > 30000:
					print('prec: ' + str(prec_1))
					precList_1.remove(prec_1)
			photList_1 = photArray_1[:, index_4].tolist()
			for phot_1 in photList_1:
				if phot_1 > 240:
					print('phot: ' + str(phot_1))
					photList_1.remove(phot_1)
			meanTempArray_1[index_1] = np.mean(np.array(tempList_1))	# 该时间段内各站点的平均气温
			sumTempArray_1[index_1] = np.sum(np.array(tempList_1))	# 该时间段内各站点的积温
			sumTempArray_2[index_1] = np.sum(np.array(tempList_2))	# 该时间段内各站点的低于极端气候阈值的积温
			sumTempArray_3[index_1] = np.sum(np.array(tempList_3))	# 该时间段内各站点的超过极端气候阈值的积温
			sumPrecArray_1[index_1] = np.sum(np.array(precList_1))	# 该时间段内各站点的平均降水
			meanPhotArray_1[index_1] = np.mean(np.array(photList_1))
			index_1 = index_1 + 1
	meanList_1 = []
	stdList_1 = []
	meanList_1.append(np.mean(meanTempArray_1))
	meanList_1.append(np.mean(sumTempArray_1))
	meanList_1.append(np.mean(sumTempArray_2))
	meanList_1.append(np.mean(sumTempArray_3))
	meanList_1.append(np.mean(sumPrecArray_1))
	meanList_1.append(np.mean(meanPhotArray_1))
	stdList_1.append(np.std(meanTempArray_1))
	stdList_1.append(np.std(sumTempArray_1))
	stdList_1.append(np.std(sumTempArray_2))
	stdList_1.append(np.std(sumTempArray_3))
	stdList_1.append(np.std(sumPrecArray_1))
	stdList_1.append(np.std(meanPhotArray_1))
	DrawStationBox_1(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, meanList_1, stdList_1, extremeList_1, tags_1, color_1, xlabel_1, outPtah_1, textLocationlist_1, figureList_1)

# 绘制站点数据散点图
def DrawStationPlot(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, outPtah_1):
	dataFrame_1 = pandas.read_csv(inPath_1)	# 读取站点表格
	dataFrame_1.sort_values('stationID', ascending = 1, inplace = True)	# 将站点表格按站点ID排序
	dataFrame_2 = dataFrame_1[(dataFrame_1['latitude'] >= extent_1[0] * 100) & (dataFrame_1['latitude'] <= extent_1[2] * 100) \
	& (dataFrame_1['longitude'] >= extent_1[1] * 100) & (dataFrame_1['longitude'] <= extent_1[3] * 100)]	# 选取在研究区范围内的记录
	stationIDList_1 = dataFrame_2['stationID'].tolist()	# 站点ID列表
	stationCount_1 = len(stationIDList_1)	# 站点ID列表长度
	dateList_1 = []
	chillingCountList_1 = []
	forcingCountList_1 = []
	for date_1 in range(dateRange_1[0], dateRange_1[1] + 1):
		if date_1 != 0:
			dateList_1.append(date_1)
			chillingCount_1 = 0
			forcingCount_1 = 0
			for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
				if date_1 < 0:
					date_2 = datetime.datetime(year_1 - 1, 1, 1) + datetime.timedelta(366 + date_1)
				elif date_1 > 0:
					date_2 = datetime.datetime(year_1, 1, 1) + datetime.timedelta(date_1 - 1)
				filename_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001-' + str(date_2.year) + str(date_2.month).zfill(2) + '.csv'	# 气温数据文件名
				inPath_4 = os.path.join(inPath_2, filename_1)
				# print(date_2)
				dataFrame_3 = pandas.read_csv(inPath_4)
				dataFrame_3.columns = ['stationID', 'latitude', 'longitude', 'altitude', 'year', 'month', 'day', 'tem_mean', 'tem_maximum', 'tem_minimum', 'tem_mean_QC', 'tem_maximum_QC', 'tem_minimum_QC']	# 修改气温数据列名
				dataFrame_4 = dataFrame_3[(dataFrame_3['day'] == date_2.day) & (dataFrame_3['stationID'].isin(stationIDList_1))]
				tempArray_1 = dataFrame_4['tem_mean'].to_numpy()	# 将气温数据写入数组
				if date_1 < 0:
					inPath_5 = os.path.join(inPath_3, str(366 + date_1) + '.csv')	# 该日极端气候阈值文件路径
				elif date_1 > 0:
					inPath_5 = os.path.join(inPath_3, str(date_1) + '.csv')	# 该日极端气候阈值文件路径
				dataFrame_5 = pandas.read_csv(inPath_5)	# 该日极端气候阈值文件
				for index_1 in range(stationCount_1):	# 遍历该日每个站点H:\2_个人\2_phenology\figure\python
					station_1 = dataFrame_4['stationID'].tolist()[index_1]	# 获取站点ID
					threshold_1 = dataFrame_5[dataFrame_5['stationID'] == station_1]['temp_10'].to_numpy()[0]	# 获取该站点的阈值
					threshold_2 = dataFrame_5[dataFrame_5['stationID'] == station_1]['temp_85'].to_numpy()[0]	# 获取该站点的阈值
					if tempArray_1[index_1] <= threshold_1:
						chillingCount_1 = chillingCount_1 + 1
					if tempArray_1[index_1] >= threshold_2:
						forcingCount_1 = forcingCount_1 + 1
			chillingCountList_1.append(chillingCount_1)
			forcingCountList_1.append(forcingCount_1)
			print(date_1)
	dataFrame_6 = pandas.DataFrame({'date': dateList_1, 'chillingCount': chillingCountList_1, 'forcingCount' : forcingCountList_1})
	dataFrame_6.to_csv(outPtah_1, index = False)


# 绘制栅格数据线盒图（未标准化）
def DrawRasterBox_1(inPath_1, range_1, extent_1, resolution_1, noData_1, mean_1, std_1, tags_1, extreme_1, color_1, xlabel_1, ylabel_1, outPtah_1):
	valueList_1 = []
	valueList_2 = []
	test = []
	for dirPath, dirname, filenames in os.walk(inPath_1):
		for filename in filenames:
			if filename[-4: ] == '.tif' and int(filename[0: 4]) >= range_1[0] and int(filename[0: 4]) <= range_1[1]:
				inPath_2 = os.path.join(inPath_1, filename)
				raster_1 = gdal.Open(inPath_2)
				array_1 = raster_1.ReadAsArray()
				valueList_3 = []
				if resolution_1 == 0.5:
					for row_1 in range(int((89.75 - extent_1[2]) / 0.5), int((89.75 - extent_1[0]) / 0.5) + 1):
						for column_1 in range(int(extent_1[1] / 0.5) - 1, int(extent_1[3] / 0.5)):
							if array_1[row_1][column_1] > noData_1:
								value_1 = (array_1[row_1][column_1] - mean_1) / std_1
								valueList_3.append(value_1)
				elif resolution_1 == 0.05:
					for row_1 in range(int((90 - extent_1[2]) / 0.05), int((90 - extent_1[0]) / 0.05) + 1):
						for column_1 in range(3600 + int(extent_1[1] / 0.05) - 1, 3600 + int(extent_1[1] / 0.05)):
							if array_1[row_1][column_1] > noData_1:
								value_1 = (array_1[row_1][column_1] - mean_1) / 1
								valueList_3.append(value_1)
				valueArray_1 = np.array(valueList_3)
				valueList_2.append(np.median(valueArray_1))
				test.append(np.mean(valueArray_1))
				valueList_1.append(valueList_3)
	print(test)
	xList_1 = []
	labels_1 = []
	index_1 = 0
	for year in range(range_1[0], range_1[1] + 1):
		xList_1.append(index_1 + 1)
		labels_1.append(str(year))
		index_1 = index_1 + 1
	plt.figure(figsize = (13, 5))
	plt.subplots_adjust(left = 0.06, bottom = 0.18, right = 0.98, top = 0.9)	

	yList_1 = []
	for valueList_2 in valueList_1:
		yList_1.append(np.median(np.array(valueList_2)))
	plt.scatter(xList_1, yList_1, c = 'black', s = 6, zorder = 3)
	fit_1 = np.polyfit(xList_1, yList_1, 1)
	ld_1 = np.poly1d(fit_1)
	pylab.plot(xList_1, ld_1(xList_1), c = 'black', zorder = 4)
	slopeVariable_1, interceptVariable_1, rValueVariable_1, pValueVariable_1, stdErrVariable_1 = scipy.stats.linregress(xList_1, yList_1)
	function_1 = 'y=' + str(ld_1).replace(' ', '') + ', R²=' + str(round(rValueVariable_1 * rValueVariable_1, 2))
	if pValueVariable_1 < 0.05:
		function_2 = function_1 + ', P<0.05'
	else:
		function_2 = function_1 + ', P≥0.05'
	plt.text(2, -30, function_2.replace('\n', ''), fontsize = 16)

	boxplot_1 = plt.boxplot(valueList_1, labels = labels_1, showfliers = False, patch_artist = True, zorder = 2)
	plt.setp(boxplot_1['medians'], color = 'black')
	medianList_1 = []
	for medianLine_1 in boxplot_1['medians']:
		median_1 = medianLine_1.get_ydata()[0]
		medianList_1.append(median_1)
	# tag_1 = range_1[0]
	# for patch in boxplot_1['boxes']:
	# 	if tag_1 in tags_1:
	# 		patch.set(facecolor = color_1)
	# 		pass
	# 	else:
	# 		patch.set(facecolor = 'lightgrey')
	# 	tag_1 =  tag_1 + 1
	tag_1 = 0
	for patch in boxplot_1['boxes']:
		if medianList_1[tag_1] > extreme_1:
			patch.set(facecolor = color_1)
		elif medianList_1[tag_1] < -extreme_1:
			pass
		else:
			patch.set(facecolor = 'lightgrey')
		tag_1 =  tag_1 + 1
	plt.axhline(y = -extreme_1, c = 'grey', ls = 'dashed', zorder = 1)
	plt.axhline(y = 0, c = 'grey', ls = 'dashed', zorder = 1)
	plt.axhline(y = extreme_1, c = 'grey', ls = 'dashed', zorder = 1)
	plt.title(label = ylabel_1, fontsize = 24)
	plt.xlabel(xlabel_1, fontsize = 18)
	# plt.ylabel(ylabel_1, fontsize = 16)
	plt.xticks(fontsize = 16, rotation = 45)
	plt.yticks(list(plt.yticks()[0]) + [extreme_1, -extreme_1], fontsize = 16)
	outPtah_2 = os.path.join(outPtah_1, '10_' + ylabel_1 + '.png')
	outPtah_3 = os.path.join(outPtah_1, '10_' + ylabel_1 + '.pdf')
	plt.savefig(outPtah_2, dpi = 330)
	plt.savefig(outPtah_3, dpi = 330)
	# plt.show()
	return valueList_2

# 绘制栅格数据线盒图（标准化）
def DrawRasterBox_2(inPath_1, range_1, extent_1, resolution_1, noData_1, tags_1, extreme_1, color_1, xlabel_1, ylabel_1, outPtah_1):
	valueList_1 = []
	for dirPath, dirname, filenames in os.walk(inPath_1):
		for filename in filenames:
			if filename[-4: ] == '.tif' and int(filename[0: 4]) >= range_1[0] and int(filename[0: 4]) <= range_1[1]:
				inPath_2 = os.path.join(inPath_1, filename)
				raster_1 = gdal.Open(inPath_2)
				array_1 = raster_1.ReadAsArray()
				if resolution_1 == 0.5:
					for row_1 in range(int((89.75 - extent_1[2]) / 0.5), int((89.75 - extent_1[0]) / 0.5) + 1):
						for column_1 in range(int(extent_1[1] / 0.5) - 1, int(extent_1[3] / 0.5)):
							if array_1[row_1][column_1] > noData_1:
								valueList_1.append(array_1[row_1][column_1])
				elif resolution_1 == 0.05:
					for row_1 in range(int((90 - extent_1[2]) / 0.05), int((90 - extent_1[0]) / 0.05) + 1):
						for column_1 in range(3600 + int(extent_1[1] / 0.05) - 1, 3600 + int(extent_1[1] / 0.05)):
							if array_1[row_1][column_1] > noData_1:
								valueList_1.append(array_1[row_1][column_1])
	valueArray_1 = np.array(valueList_1) 
	mean_1 = np.mean(valueArray_1) 
	std_1 = np.std(valueArray_1)
	DrawRasterBox_1(inPath_1, range_1, extent_1, resolution_1, noData_1, mean_1, std_1, tags_1, extreme_1, color_1, xlabel_1, ylabel_1, outPtah_1)

# 绘制细节散点图
def DrawItemsPlot(inPath_1, variableList_1, labelList_1, outPtah_1):
	dataFrame_1 = pandas.read_csv(inPath_1)
	phenArray_1 = dataFrame_1['phen'].to_numpy()
	phenArray_2 = preprocessing.StandardScaler().fit_transform(phenArray_1.reshape(-1, 1))
	dataFrame_1['phen_nom'] = phenArray_2.reshape(1, -1)[0]
	index_1 = 0
	for variable_1 in variableList_1:
		variableArray_1 = dataFrame_1[variable_1].to_numpy()
		variableArray_2 = preprocessing.StandardScaler().fit_transform(variableArray_1.reshape(-1, 1))
		dataFrame_1[variable_1 + '_nom'] = variableArray_2.reshape(1, -1)[0]
		for landcover_1 in ['Forest', 'Grassland']:
			dataFrame_2 = dataFrame_1[(dataFrame_1['range'] == 30) & (dataFrame_1['landcover'] == landcover_1)]
			phenArray_3 = dataFrame_2['phen_nom'].to_numpy()
			variableArray_3 = dataFrame_2[variable_1 + '_nom'].to_numpy()
			slopeVariable_1, interceptVariable_1, rValueVariable_1, pValueVariable_1, stdErrVariable_1 = scipy.stats.linregress(variableArray_3.tolist(), phenArray_3.tolist())
			function_1 = variable_1 + '-' + landcover_1 + ': y=' + str(round(slopeVariable_1, 4)) + 'x+' + str(round(interceptVariable_1, 2)) + ', R=' + str(round(rValueVariable_1, 2))
			if pValueVariable_1 < 0.05:
				function_2 = function_1 + ', P<0.05'
			else:
				function_2 = function_1 + ', P≥0.05'
			print(function_2)
		p = ggplot(dataFrame_1[(dataFrame_1['range'] == 30) & (dataFrame_1['landcover'] != 'Artificial surfaces')], aes(x = variable_1 + '_nom', y = 'phen_nom')) + geom_point(size = 0.5) + \
		geom_smooth(method = 'lm', size = 0.5) + facet_wrap('landcover') + theme_light() + labs(x = labelList_1[index_1], y = 'Start of season') + \
		theme(axis_title = element_text(size = 18, family = 'serif'), axis_text_x = element_text(size = 16, family = 'serif'), axis_text_y = element_text(size = 16, family = 'serif'), strip_text = element_text(size = 18, color = 'black', family = 'serif'))
		outPtah_2 = os.path.join(outPtah_1, str(index_1 + 10) + '_' + labelList_1[index_1] + '.jpg')
		outPtah_3 = os.path.join(outPtah_1, str(index_1 + 10) + '_' + labelList_1[index_1] + '.pdf')
		ggsave(p, outPtah_2, width = 15, height = 4, dpi = 330)
		ggsave(p, outPtah_3, width = 15, height = 4, dpi = 330)
		index_1 = index_1 + 1

# 绘制结果散点图
def DrawResultPlot(inPath_1, variableList_1, labelList_1, outPtah_1):
	dataFrame_1 = pandas.read_csv(inPath_1)
	index_1 = 0
	for variable_1 in variableList_1:
		dataFrame_2 = dataFrame_1[(dataFrame_1['landcover'] != 'Cultivated land') & (dataFrame_1['landcover'] != 'Artificial surfaces') & ((dataFrame_1['type'] == variable_1) | (dataFrame_1['type'] == variable_1.replace('Coef', 'Vip', 1)))]
		p = ggplot(dataFrame_2) + geom_line(aes(x = 'range', y = 'value', color = 'type')) + geom_hline(aes(yintercept = 0.8), linetype = 'dashed', size = 0.3) + geom_hline(aes(yintercept = 1), linetype = 'dashed', size = 0.3) + \
		facet_wrap('landcover') + scale_color_manual(labels = ['coefficient', 'VIP'], values = ['red', 'blue']) + labs(x = 'Time window', y = labelList_1[index_1]) + theme_light() + \
		theme(axis_title = element_text(size = 18, family = 'serif'), axis_text_x = element_text(size = 16, family = 'serif'), axis_text_y = element_text(size = 16, family = 'serif'), strip_text = element_text(size = 18, color = 'black', family = 'serif'), \
			legend_title = element_blank(), legend_position = (0.8, 0.01), legend_direction = 'horizontal', legend_text = element_text(size = 16, family = 'serif'))
		outPtah_2 = os.path.join(outPtah_1, str(index_1 + 15) + '_' + labelList_1[index_1] + '.jpg')
		outPtah_3 = os.path.join(outPtah_1, str(index_1 + 15) + '_' + labelList_1[index_1] + '.pdf')
		ggsave(p, outPtah_2, width = 15, height = 4, dpi = 330)
		ggsave(p, outPtah_3, width = 15, height = 4, dpi = 330) 
		index_1 = index_1 + 1

def main():
	# # 绘制气象要素线盒图
	# variable_1 = 'SURF_CLI_CHN_MUL_DAY-TEM-12001'
	# inPath_1 = r'H:\2_个人\2_phenology\data\5_observation\0_base\station_lc_3.csv'
	# inPath_2 = os.path.join(r'H:\2_个人\2_phenology\data\5_observation\3_select')  
	# inPath_3 = r'H:\2_个人\2_phenology\data\5_observation\5_threshold\10_85'
	# outPtah_1 = os.path.join(r'H:\2_个人\2_phenology\figure\python')  
	# textLocationlist_1 = [[20, 5], [20, 800], [20, 250], [5, 600], [20, 200], [20, -3], [20, 25], [5, 70]]
	# figureList_1 = ['Departure of mean temperature (℃)', 'Departure of accumulated temperature (d·℃)', 'Departure of chilling index (d·℃)', 'Departure of forcing index (d·℃)', \
	# 'Departure of total precipitation (100 mm)', 'Departure of mean photoperiod (h)', 'Chilling frequency (times)', 'Forcing frequency (times)']
	# yearRange_1 = [1987, 2016]
	# dateRange_1 = [-32, 151]
	# # dateRange_1 = [-5, 5]
	# extent_1 = [32, 105, 35, 114]
	# extremeList_1 = [1, 200, 200, 200, 50, 1]
	# tags_1 = [2000, 2007, 2008, 2010, 2011, 2015]
	# color_1 = 'red'
	# xlabel_1 = 'Year'
	# meanList_1 = []
	# stdList_1 = []
	# DrawStationBox_2(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, extremeList_1, tags_1, color_1, xlabel_1, outPtah_1, textLocationlist_1, figureList_1)
	# # DrawStationBox_1(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, meanList_1, stdList_1, extremeList_1, tags_1, color_1, xlabel_1, outPtah_1, figureList_1)

	# # 绘制站点数据散点图
	# inPath_1 = r'H:\2_个人\2_phenology\data\5_observation\0_base\station_lc_3.csv'
	# inPath_2 = os.path.join(r'H:\2_个人\2_phenology\data\5_observation\3_select')  
	# inPath_3 = r'H:\2_个人\2_phenology\data\5_observation\5_threshold'
	# outPtah_1 = os.path.join(r'H:\2_个人\2_phenology\figure')  
	# yearRange_1 = [1987, 2016]
	# dateRange_1 = [-32, 151]
	# item_1 = 'Qinling_' + str(yearRange_1[0]) + '-' + str(yearRange_1[1]) + '_' + str(dateRange_1[0]) + '-' + str(dateRange_1[1])
	# extent_1 = [32, 105, 35, 114]
	# outPtah_1 = os.path.join(r'H:\2_个人\2_phenology\data\6_result', 'extreme_' + item_1 + '.csv')
	# DrawStationPlot(inPath_1, inPath_2, inPath_3, yearRange_1, dateRange_1, extent_1, outPtah_1)

	# # 绘制物候参数线盒图
	# inPath_1 = r'J:\2_personal\1_papers\2_phenology\data\3_phenology\0_Start of Season 1'
	# range_1 = [1987, 2016]
	# extent_1 = [32, 105, 35, 114]
	# # extent_1 = [47, 117, 54, 128]	# 东北
	# resolution_1 = 0.05
	# noData_1 = -1
	# tags_1 = [2000, 2007, 2008, 2010, 2011, 2015]
	# extreme_1 = 5
	# color_1 = 'red'
	# xlabel_1 = 'Year'
	# ylabel_1 = 'Departure of start of season (days)'
	# outPtah_1 = r'J:\2_personal\1_papers\2_phenology\figure\python'
	# DrawRasterBox_2(inPath_1, range_1, extent_1, resolution_1, noData_1, tags_1, extreme_1, color_1, xlabel_1, ylabel_1, outPtah_1)

	# # # 绘制细节散点图
	# inPath_1 = r'H:\2_个人\2_phenology\data\6_result\秦岭\1987-2016_-32-151\items_Qinling_1987-2016_-32-151_landcover.csv'
	# variableList_1 = ['temp_mean', 'temp_10', 'temp_85', 'prec_sum', 'phot_mean']
	# labelList_1 = ['Mean temperature', 'Chilling index', 'Forcing index', 'Total precipitation', 'Mean photoperiod']
	# outPtah_1 = r'H:\2_个人\2_phenology\figure\python'
	# DrawItemsPlot(inPath_1, variableList_1, labelList_1, outPtah_1)

	# # 绘制结果散点图
	# inPath_1 = r'H:\2_个人\2_phenology\data\6_result\秦岭\1987-2016_-32-151\result_Qinling_1987-2016_-32-151_long_landcover.csv'
	# variableList_1 = ['temp_meanCoef', 'temp_10Coef', 'temp_85Coef', 'prec_sumCoef', 'phot_meanCoef']
	# labelList_1 = ['Contribution of mean temperature', 'Contribution of chilling', 'Contribution of forcing', 'Contribution of total precipitation', 'Contribution of mean photoperiod']
	# outPtah_1 = r'H:\2_个人\2_phenology\figure\python'
	# DrawResultPlot(inPath_1, variableList_1, labelList_1, outPtah_1)

	# dataFrame_1 = pandas.read_csv(r'J:\2_personal\1_papers\2_phenology\revision_1\0_data\1_phenology\4_observation\站点数据_回归.csv')
	# xList_1 = dataFrame_1['Year'].tolist()
	# yList_1 = dataFrame_1['Satellite-retrieved SOS'].tolist()
	# slopeVariable_1, interceptVariable_1, rValueVariable_1, pValueVariable_1, stdErrVariable_1 = scipy.stats.linregress(xList_1, yList_1)
	# print(slopeVariable_1, rValueVariable_1, pValueVariable_1)

	# dataFrame_1 = pandas.read_csv(r'J:\2_personal\1_papers\2_phenology\data\6_result\秦岭\1987-2016_-32-151\result_Qinling_1987-2016_-32-151_long_landcover.csv')
	# outPtah_1 = r'J:\2_personal\1_papers\2_phenology\revision_1\0_data\3_result'
	# for lc in ['Cultivated land', 'Grassland', 'Forest']:
	# 	dataFrame_2 = dataFrame_1[dataFrame_1['landcover'] == lc]
	# 	for cf in ['phot_meanVip', 'prec_sumVip', 'temp_10Vip', 'temp_85Vip', 'temp_meanVip', 'temp_sumVip']:
	# 		dataFrame_3 = dataFrame_2[dataFrame_2['type'] == cf]
	# 		outPtah_2 = os.path.join(outPtah_1, lc + '_' + cf + '.csv')
	# 		dataFrame_3.to_csv(outPtah_2, index = False)

	strt_date = date(int(2009), 1, 2)
	for doy in range(-26, 149, 6):
		end_date = strt_date + timedelta(days = doy)
		print(doy, end_date)


	strt_date = date(int(2009), 1, 2)
	for doy in range(-32, 143, 6):
		end_date = strt_date + timedelta(days = doy)
		print(doy, end_date)

main()