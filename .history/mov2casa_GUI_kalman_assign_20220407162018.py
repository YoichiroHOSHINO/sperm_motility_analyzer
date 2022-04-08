# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from itertools import chain
from PIL import Image, ImageDraw
from PIL import ImageFont
import math
import time
from scipy.optimize import linear_sum_assignment

# メイン

def main(movie_files_path, result_dir, config_df, TestMode):
    SetArg(config_df)
    global MovieFolder
    global ResultFolder
    global filename_with_arg
    ResultFolder = result_dir + '/'
    filename_with_arg = ''
    
    if TestMode == 0:
        doAnalysis(movie_files_path)
    else:
        makeTestImg(movie_files_path)
    
    cv2.destroyAllWindows()

def doAnalysis(movie_files_path):
    totalfilenum = len(movie_files_path)
    filenum = 1
    AllResults_filename = ResultFolder + 'AllResults' + "_from" + str(start_frame) + "to" + str(end_frame) + 'calc' + str(max_frame) + 'frames.csv'
    if os.path.exists(AllResults_filename):
        AllResults_df_exist = pd.read_csv(AllResults_filename)
        AllResults = AllResults_df_exist.values.tolist()
    else:
        AllResults = []

    ALLRF = pd.DataFrame()
    for filepath in tqdm(movie_files_path):      # メインルーチン
        start_time = time.perf_counter()
        f = os.path.basename(filepath)
        global filename_with_arg
        filename_with_arg = f + "_from" + str(start_frame) + "to" + str(end_frame) + 'calc' + str(max_frame) + '_frames_'
          
        print (f + ' を処理しています。(' + str(filenum) + '/' + str(totalfilenum) + ')')
        filenum +=1

        moviearray, movieBWarray = makemovarray(filepath)
        maskarray = makeMask(movieBWarray) # 不動精子マスクを作成
        #movBWarray = makemovBWarray(movieBWarray, maskarray) #(movarray, mask, dark_erosion_iter, dark_dilate_iter, AThreshBS, AThreshC)
        df = findParticleFromMov(movieBWarray, maskarray)   # 動画から粒子検出
        
        h = ['index','connect_index','Frame','x','y','area','pre_x','pre_y','Point','Ave_x','Ave_y','pred_x','pred_y','pred_vx','pred_vy'
            ,'Length','Runlength','Ave_Length','Ave_RunLength','Framelength','Velocity','angle','fix_past','fix_next','motile']
        pnt = 0
        df, pnt = makeTracks(df, pnt, h)
        #zarray = makezarray(FrameRate)
        #df = fixImInZero(df, pnt, FrameRate, zarray, h)

        pandasdf = pd.DataFrame(df, columns=h)
        dfs = pandasdf.sort_values(by=['Point','Frame'], ascending=True)
                
        dfck = np.where(df[:,h.index('fix_past')] > 0)
        if dfck[0].shape[0] > 0:

            VCL_VAP = makeVCL_VAP(df, h)
            #VAP = makeVAP(df, microscale, h)
            VSL = makeVSL(df, microscale, h)
            
            BCF = makeBCF(df, h)
            ALH = makeALH(df, microscale, h)

            FRD = makeFRD(df, microscale, h)
            ANG = makeANG(df, h)

            #RF = pd.merge(VCL, VAP, on="Point")
            #RF = pd.merge(RF, VSL.loc[:, ["Point", "VSL"]], on="Point")
            RF = pd.merge(VCL_VAP, VSL.loc[:, ["Point", "VSL"]], on="Point")
            RF = pd.merge(RF, BCF, on='Point')
            RF = pd.merge(RF, ALH, on='Point')
            RF["LIN"] = RF.VSL / RF.VCL
            RF["WOB"] = RF.VAP / RF.VCL 
            RF["STR"] = RF.VSL / RF.VAP
            RF = pd.merge(RF, ANG, on='Point')
            RF = pd.merge(RF, FRD, on='Point')
        else:
            RF = pd.DataFrame([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]],columns=['Point','Mov','FL_VCL','VCL','FL_VAP','VAP','VSL','BCF','ALH','LIN','WOB','STR','SumAngle','MeanAngle','StdAngle','diameter','D'])

        RF.fillna(0, inplace = True)
        RF, dfs = Add_decision_simple(RF, dfs, Motile_thresh_diameter)
        RF, dfs = Add_decision_derail(RF, dfs, Derail_StdAngle=100)

        frames = 60
        
        RF, dfs = Add_decision_prog(RF, dfs, frames, Progressive)
        RF, dfs = Add_decision_circle(RF, dfs, frames, Circle_SumAngle, Circle_MeanAngle, Circle_StdAngle)

        RF.to_csv(ResultFolder + f +  "_sec" + str(start_frame)  + "_CASA.csv", index=False)
        dfs.to_csv(ResultFolder + f +  "_sec" + str(start_frame)  + "_AllPoints.csv", index=False)

        Zeroall = dfs[dfs.Frame == 0]
        total = Zeroall.shape[0]
        motile = Zeroall['motile'].sum()
        Zeroprogs = Zeroall['motile']*Zeroall['prog']
        prog = Zeroprogs.sum()
        Zerocircles = Zeroall['motile']*Zeroall['circle']
        circle = Zerocircles.sum()

        motility = motile/total
        prograte = prog/total
        circlerate = circle/total

        RFhigh = RF[(RF.FL_VCL == frames-1)&(RF.motility == 1)].copy()

        meanRFhigh = RFhigh.median()
        No_motil = RFhigh.shape[0]
        tmpdf = pd.DataFrame([[f + "_sec" + str(start_frame),
                                No_motil,
                                meanRFhigh["VCL"],
                                meanRFhigh["VAP"],
                                meanRFhigh["VSL"],
                                meanRFhigh["BCF"],
                                meanRFhigh["LIN"],
                                meanRFhigh["WOB"],
                                meanRFhigh["STR"],
                                meanRFhigh["ALH"],
                                meanRFhigh["SumAngle"],
                                meanRFhigh["diameter"],
                                meanRFhigh["D"],
                                total,
                                motility,
                                prograte,
                                circlerate]],
                                    columns=["File",
                                            "No. Analyzed",
                                            "VCL",
                                            "VAP",
                                            "VSL",
                                            "BCF",
                                            "LIN",
                                            "WOB",
                                            "STR",
                                            "ALH",
                                            "angle",
                                            "diameter",
                                            "D",
                                            "0F_total",
                                            "Motility",
                                            "Prog_Rate",
                                            "Circle_Rate"])
        SaveAllResults(tmpdf, ResultFolder)

        writeResultMovFast(ResultFolder, f, moviearray, movieBWarray, dfs, start_frame, 0, h)
        #writeResultMovFast(MovieFolder, ResultFolder, f, movarray, movBWarray, dfs, start_second, 1)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print ('処理時間：' + str(elapsed_time))

# 認識テスト画像生成
def makeTestImg(movie_files_path):
    totalfilenum = len(movie_files_path)
    filenum = 1

    for filepath in movie_files_path:       # メインルーチン
        filename = os.path.basename(filepath)
        print (filename + ' の画像認識テストを出力します。(' + str(filenum) + '/' + str(totalfilenum) + ')')
        filenum +=1
        cap = cv2.VideoCapture(filepath)  #MovieFolder + 
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameend = full_frames
        if end_frame > 0 and end_frame < full_frames:
            frameend = end_frame

        framestart = start_frame

        for f in range(frameend):
            if f < framestart:
                ret, img = cap.read()
            else:
                movimg = readimg(cap, height, width)
                BWimg, dark, bright = nichika(movimg)

                pnt = 0

                df, pnt, conts = findParticleZero(BWimg, pnt, f)

                BWframe = cv2.merge((BWimg, BWimg, BWimg))
                original = cv2.merge((movimg, movimg, movimg))

                for k in range(df.shape[0]): #df.index:
                    BWframe = cv2.circle(BWframe,(int(df[k,1]), int(df[k,2])),3,(0,0,255),-1)

                original_withDet = cv2.drawContours(original, conts, -1, (255,255,0),1)
                for k in range(df.shape[0]): #df.index:
                    Dtframe = cv2.circle(original_withDet,(int(df[k,1]), int(df[k,2])),3,(0,0,255),-1)
        
                cv2.imwrite(ResultFolder + filename + "_raw.jpg", movimg)
                cv2.imwrite(ResultFolder + filename + "_Detect.jpg", Dtframe)
                cv2.imwrite(ResultFolder + filename + "_BWdetect.jpg", BWframe)
                cv2.imwrite(ResultFolder + filename + "_dark.jpg", dark)
                cv2.imwrite(ResultFolder + filename + "_bright.jpg", bright)

                #cv2.imshow(Dtframe)

                break

        df = AllPoints
        BWimg = movBWarray[0]

        orig = movarray[0]
        BWframe = cv2.merge((BWimg, BWimg, BWimg))
        original = cv2.merge((orig, orig, orig))

        for k in range(df.shape[0]): #df.index:
            BWframe = cv2.circle(BWframe,(int(df[k,1]), int(df[k,2])),3,(0,0,255),-1)
        
        frame2 = cv2.hconcat([original, BWframe])

        cv2.imwrite(ResultFolder + filename + "_detect.jpg", frame2)
        cv2.imwrite(ResultFolder + filename + "_mask.jpg", mask)
        cv2.imwrite(ResultFolder + filename + "_dark.jpg", dark_dtct)
        cv2.imwrite(ResultFolder + filename + "_bright.jpg", bright_dtct)

    return 0

def makemovarray(filepath):
    print ('解析用画像スタックを作成しています')
    cap = cv2.VideoCapture(filepath)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    array = []
    BWarray = []
    for f in tqdm(range(end_frame)):
        ret, img = cap.read()
        if f >= start_frame:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = crop(img, height, width, cropheight, cropwidth)
            BWimg, dark, bright = nichika(img)
            array.append(img)
            BWarray.append(BWimg)
    
    moviearray = np.array(array, dtype='uint8')
    movieBWarray = np.array(BWarray, dtype='uint8')

    return moviearray, movieBWarray

def makeMask(movieBWarray):
    MaskThreshold = 250
    print ('不動精子マスクを作成しています')
    frames = movieBWarray.shape[0]

    maskarray = []

    for f in range(frames-avearea):
        pile = np.zeros((movieBWarray.shape[1],movieBWarray.shape[2]))
        for i in range(avearea):
            img = movieBWarray[f+i]/avearea
            pile = pile + img
        ret, mask = cv2.threshold(pile,MaskThreshold,255,cv2.THRESH_BINARY)
        maskarray.append(mask)
    
    for t in range(avearea):
        maskarray.append(mask)
    
    maskarray = np.array(maskarray, dtype='uint8')

    return maskarray

# 画像二値化処理
def nichika(img):
    # 明部検出
    bright_erosion_iter = 0
    bright_dilate_iter = 1
    ret, thresh1 = cv2.threshold(img,Bright_thresh,255,cv2.THRESH_BINARY)
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
    thresh1 = cv2.erode(thresh1, kernel, iterations = bright_erosion_iter)
    thresh1 = cv2.dilate(thresh1, kernel, iterations = bright_dilate_iter)

    if Threshtype == 1:
        thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,AThreshBS,AThreshC)  # 暗部検出：通常の適応的二値化
    elif Threshtype == 0:
        ret, thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        ret, thresh2 = cv2.threshold(img,Threshtype,255,cv2.THRESH_BINARY)
    
    if cv2.mean(thresh2)[0] > 128:
        thresh2 = ~thresh2  # 白黒反転
    
    thresh2 = cv2.erode(thresh2, kernel, iterations = dark_erosion_iter)
    thresh2 = cv2.dilate(thresh2, kernel, iterations = dark_dilate_iter)

    thresh3 = thresh2 + thresh1

    img = thresh3 #.astype(np.uint8)
    
    return img, thresh2, thresh1

def makesaveheader(heads):
    hdr = ""
    for h in heads:
        hdr += h + ',' 
    hdr = hdr[:-1]
    return hdr

# 変数読み込み
def SetArg(cf):
    global cnt_analysis
    global cropheight
    global cropwidth
    global FrameRate
    global start_frame
    global max_frame
    global end_frame
    global minsize
    global maxsize
    global ovalratio
    global mobsearchrange
    global maxjumpframes
    global microscale
    global Threshtype
    global AThreshBS
    global AThreshC
    global Bright_thresh
    global bright_erosion_iter
    global bright_dilate_iter
    global dark_erosion_iter
    global dark_dilate_iter
    global Motile_thresh_VSL
    global Motile_thresh_diameter
    global Progressive
    global Circle_SumAngle
    global Circle_MeanAngle
    global Circle_StdAngle
    global avearea

    cnt_analysis = int(cf.loc[cf[0] == "cnt_analysis",1].values[0])
    cropheight = int(cf.loc[cf[0] == "cropheight",1].values[0])
    cropwidth = int(cf.loc[cf[0] == "cropwidth",1].values[0])
    FrameRate = int(cf.loc[cf[0] == "FrameRate",1].values[0])
    start_frame = int(cf.loc[cf[0] == "start_frame",1].values[0])
    max_frame = int(cf.loc[cf[0] == "max_frame",1].values[0])
    end_frame = int(cf.loc[cf[0] == "end_frame",1].values[0])
    minsize = int(cf.loc[cf[0] == "minsize",1].values[0])
    maxsize = int(cf.loc[cf[0] == "maxsize",1].values[0])
    ovalratio = float(cf.loc[cf[0] == "ovalratio",1].values[0])
    mobsearchrange = int(cf.loc[cf[0] == "mobsearchrange",1].values[0])
    maxjumpframes = int(cf.loc[cf[0] == "maxjumpframes",1].values[0])
    microscale = float(cf.loc[cf[0] == "microscale",1].values[0])
    Threshtype = int(cf.loc[cf[0] == "Threshtype",1].values[0])
    Threshtype = int(cf.loc[cf[0] == "Threshtype",1].values[0])
    AThreshBS = int(cf.loc[cf[0] == "AThreshBS",1].values[0])
    AThreshC = int(cf.loc[cf[0] == "AThreshC",1].values[0])
    Bright_thresh = int(cf.loc[cf[0] == "Bright_thresh",1].values[0])
    bright_erosion_iter = int(cf.loc[cf[0] == "dark_erosion_iter",1].values[0])
    bright_dilate_iter = int(cf.loc[cf[0] == "dark_dilate_iter",1].values[0])
    dark_erosion_iter = int(cf.loc[cf[0] == "dark_erosion_iter",1].values[0])
    dark_dilate_iter = int(cf.loc[cf[0] == "dark_dilate_iter",1].values[0])
    Motile_thresh_VSL = float(cf.loc[cf[0] == "Motile_thresh_VSL",1].values[0])
    Motile_thresh_diameter = float(cf.loc[cf[0] == "Motile_thresh_diameter",1].values[0])
    Progressive = int(cf.loc[cf[0] == "Progressive",1].values[0])
    Circle_SumAngle = int(cf.loc[cf[0] == "Circle_SumAngle",1].values[0])
    Circle_MeanAngle = int(cf.loc[cf[0] == "Circle_MeanAngle",1].values[0])
    Circle_StdAngle = int(cf.loc[cf[0] == "Circle_StdAngle",1].values[0])
    avearea = int(FrameRate/6)

# 動画ファイルから粒子抽出

#@jit
def findParticleFromMov(movieBWarray, maskarray):
    arlist = []
    frames = movieBWarray.shape[0] #FrameRate #int(cap.get(cv2.CAP_PROP_FPS)) + 1 # 調査フレーム数をフレームレートに設定＝1秒間当たりの動きを調べる
    #avearea = int(FrameRate/6)   # フレーム数の1/6を平均経路を求める数にする。
    print ('動画から粒子検出。')
    lastidx = 1
    for f in tqdm(range(frames)):
        for t in [0,1]:
            if t == 0:
                bimg = movieBWarray[f] - maskarray[f]
            else:
                bimg = maskarray[f]

            if int(cv2.__version__[:1]) < 4:
                image, contours, hierarchy = cv2.findContours(bimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 輪郭検出
            else:
                contours, hierarchy = cv2.findContours(bimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):  # 重心座標と簡易面積の算出
                M = cv2.moments(contours[i]) #モーメントを求める
                (x,y),radius = cv2.minEnclosingCircle(contours[i]) # 最小外接円を求める
                area = M['m00']     # 面積
                minCircleArea = radius * radius * 3.14      # 最小外接円の面積
                if area != 0:
                    if (area > minsize)&(area < maxsize)&(area < minCircleArea * ovalratio):  # 面積が最小以上、最大以下、最小外接円の面積× ovalratio以下の場合
                        cX = int(M['m10']/M['m00'])
                        cY = int(M['m01']/M['m00'])
                        #onmask = mask[cY, cX]
                        if t == 1:  #onmask > 0:
                            Mov = 0
                        else:
                            Mov = 1
                        #h = ['index','connect_index','Frame','x','y','area','pre_x','pre_y','Point','Ave_x','Ave_y','pred_x','pred_y','pred_vx','pred_vy'
                        #    ,'Length','Runlength','Ave_Length','Ave_RunLength','Framelength','Velocity','angle','fix_past','fix_next','motile']
                        arlist.append([lastidx,0,f,cX,cY,area,cX,cY,0,cX,cY,cX,cY,0,0,0,0,0,0,0,0,0,0,0,Mov])
                        lastidx += 1

    df = np.array(arlist, dtype='float16')
    return df

def crop(img, height, width, cropheight, cropwidth):
    if cropheight >= height or cropheight == 0:
        outheight = height
    else:
        outheight = cropheight
    if cropwidth >= width or cropwidth == 0:
        outwidth = width
    else:
        outwidth = cropwidth
    ystart = int(height/2) - int(outheight/2)    
    xstart = int(width/2) - int(outwidth/2)
    cropimg = img[int(ystart):int(ystart+outheight),int(xstart):int(xstart+outwidth)]

    return cropimg

################################################################
# カルマンフィルター実装
kalman = cv2.KalmanFilter(2,1,0)

def initKalman():
    kalman.statePre = np.zeros((2,1),dtype=np.float32)
    kalman.statePost = np.zeros((2,1),dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0]],np.float32)
    kalman.processNoiseCov = np.array([[0.01, 0],[0, 0.01]],np.float32)
    kalman.measurementNoiseCov = np.array([[0.1]],np.float32)
    kalman.errorCovPost = np.array([[0.1370390355587,0.0486866608262062],[0.0486866608262062,0.0381471514701843]],np.float32)
    kalman.transitionMatrix = np.array([[1,1],[0,1]],np.float32)

def Kalman(x, px, pvx):
    initKalman()
    measurement = np.zeros((1),dtype=np.float32)
    measurement[0] = x.astype(np.float32)
    pred = np.array([px.astype(np.float32), pvx.astype(np.float32)])
    kalman.statePre = pred
    kalman.statePost = pred
    estim = kalman.correct(measurement)
    pred = kalman.predict()
    #print (x, px, pvx, pred)
    px = pred[0]
    pvx = pred[1]
    return px, pvx

################################################################

# 動精子連結

def makeTracks(df, pnt, h):
    print ('軌跡を連結しています')
    meanLeng = 5

    for f in range(start_frame + 1, end_frame):
        for s in [0,1]:
            if s == 0:
                motile = 1
            else:
                motile = 0
            
            matrix_xyz = np.empty(0)
            p_idx_stack = []
            layer_stack = []
            n_idx_stack = []

            for t in range(maxjumpframes):
                if f - start_frame - t >= 1:
                    t = t + 1

                    n_idx = np.where((df[:,h.index('Frame')] == f+t)&(df[:,h.index('fix_past')] == 0)&(df[:,h.index('motile')] == motile))
                    #n_idx = np.where((df[:,h.index('Frame')] == str(f))&(df[:,h.index('fix_past')] == '0'))

                    n_idx_stack.extend(list(n_idx[0]))
                    layer_stack.extend([t]*len(n_idx[0]))
                    #n_idx_stack.extend(list(n_idx[0]))
                    
                    if t == 1:
                        p_idx = np.where((df[:,h.index('Frame')] == f)&(df[:,h.index('fix_next')] == 0)&(df[:,h.index('motile')] == motile))
                        p_idx_stack.extend(list(p_idx[0]))

                    if len(p_idx[0]) > 0 and len(n_idx[0]) > 0:
                        pxs = df[p_idx[0], h.index('x')]
                        pys = df[p_idx[0], h.index('y')]
                        nxs = df[n_idx[0], h.index('x')]
                        nys = df[n_idx[0], h.index('y')]

                        matrix_x = abs(nxs.reshape(1,len(nxs)) - pxs.reshape(len(pxs),1))
                        matrix_y = abs(nys.reshape(1,len(nys)) - pys.reshape(len(pys),1))
                        matrix_xy = matrix_x + matrix_y + (meanLeng*(t-1))
                        
                    if matrix_xyz.shape[0] == 0:
                        matrix_xyz = matrix_xy
                    else:
                        matrix_xyz = np.hstack([matrix_xyz, matrix_xy])

            p_assign_idxs, n_assign_idxs = linear_sum_assignment(matrix_xyz)

            ps = []
            ns = []
            layers = []

            for i in range(len(p_assign_idxs)):
                cost = matrix_xyz[p_assign_idxs[i]][n_assign_idxs[i]]
                if cost < mobsearchrange:
                    ps.append(p_idx_stack[p_assign_idxs[i]])
                    ns.append(n_idx_stack[n_assign_idxs[i]])
                    layers.append(layer_stack[p_assign_idxs[i]])

            for psi in ps:
                if df[psi, h.index('Point')] == 0:
                    df[psi, h.index('Point')] = pnt
                    pnt += 1
                
            px, py = df[ps, h.index('x')], df[ps, h.index('y')]
            nx, ny = df[ns, h.index('x')], df[ns, h.index('y')]
            df[ns, h.index('pre_x')] = px
            df[ns, h.index('pre_y')] = py

            minspan = np.sqrt(pow(px - nx,2) + pow(py - ny,2))
            Leng = minspan * microscale
            RL = df[ps,h.index('Runlength')] + Leng
            pre_FL = df[ps, h.index('Framelength')]
            FL = pre_FL + layers
            df[ns, h.index('Point')] = df[ps, h.index('Point')]
            df[ns, h.index('Length')] = Leng
            df[ns, h.index('Runlength')] = RL
            df[ns, h.index('Framelength')] = FL # フレーム長を加算
            velo = RL/FL
            df[ns, h.index('Velocity')] = velo # 速度を記録
            df[ps, h.index('fix_next')] = 1
            df[ns, h.index('fix_past')] = 1
            meanLeng = df[ns, h.index('Length')].mean()

            # 平均経路座標、距離算出
            for p in range(len(ps)):
                if f - start_frame + 1 < avearea:
                    avea = f - start_frame + 1
                else:
                    avea = avearea
                point = df[ps[p], h.index('Point')]
                Aveareaidx = np.where((df[:,h.index('Point')] == point)&(df[:,h.index('Frame')] <= f)&(df[:,h.index('Frame')] >= f+1-avea))
                Ave_df = df[Aveareaidx[0]]
                Ave_x = df[ps[p],h.index('Ave_x')]
                Ave_y = df[ps[p],h.index('Ave_y')]
                n_Ave_x = np.mean(Ave_df[:,h.index('x')])
                n_Ave_y = np.mean(Ave_df[:,h.index('y')])
                df[ns[p], h.index('Ave_x')] = n_Ave_x
                df[ns[p], h.index('Ave_y')] = n_Ave_y
                Ave_Length = microscale * np.sqrt(pow(Ave_x - n_Ave_x,2)+pow(Ave_y- n_Ave_y,2))
                df[ns[p], h.index('Ave_Length')] = Ave_Length
                n_Ave_RunLength = df[ps[p],h.index('Ave_RunLength')] + Ave_Length
                df[ns[p], h.index('Ave_RunLength')] = n_Ave_RunLength

            # カルマン予測実装
            pre_px, pre_py, pre_pvx, pre_pvy = df[ps, h.index('pred_x')], df[ps, h.index('pred_y')], df[ps, h.index('pred_vx')], df[ps, h.index('pred_vy')]
            px, pvx = Kalman(nx, pre_px, pre_pvx)
            py, pvy = Kalman(ny, pre_py, pre_pvy)
            df[ns, h.index('pred_x')] = px
            df[ns, h.index('pred_y')] = py
            df[ns, h.index('pred_vx')] = pvx
            df[ns, h.index('pred_vy')] = pvy
            #

    return df, pnt


# ndarray-dataframe変換
def ARtoDF(df, h):
    pandasdf = pd.DataFrame(df, columns=(makesaveheader(h)))
    dfs = pandasdf.sort_values(by=['Point','Frame'], ascending=True)

    return dfs
 
#VCL算出
def makeVCL_VAP(df, h):
    dfidx = np.where((df[:,h.index('Point')] == df[:,h.index('Point')])&(df[:,h.index('area')] > 0))
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    VCLlist = []
    print ('VCL,VAPを算出しています。')
    for i in tqdm(range(maxpoint)):
        pointdfidx = np.where(df[:,h.index('Point')] == i)
        pointdf = df[pointdfidx[0]]
        #if len(pointdf) > 0:
            #print (len(pointdf))
        maxFrame = pointdf[:,h.index('Framelength')].max()
        idx = np.where(pointdf[:,h.index('Framelength')] == maxFrame)
        RunLength = pointdf[idx[0][0],h.index('Runlength')]
        Ave_maxFrame = pointdf[:,h.index('Frame')].max()
        Ave_idx = np.where(pointdf[:,h.index('Frame')] == Ave_maxFrame)
        Ave_RunLength = pointdf[Ave_idx[0][0],h.index('Ave_RunLength')]        
        m = pointdf[0,h.index('motile')]
        VCLlist.append([i,m,maxFrame,RunLength,Ave_maxFrame,Ave_RunLength])
    VCLframe = np.array(VCLlist, dtype='float')
    VCLframe = pd.DataFrame(VCLframe, columns=["Point", "motile", "FL_VCL", "VCL", "FL_VAP", "VAP"])
    return VCLframe

def makeVSL(df, microscale, h):
    dfidx = np.where((df[:,h.index('Point')] == df[:,h.index('Point')])&(df[:,h.index('area')] > 0))
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    VSLlist = []
    print ('VSLを算出しています。')
    for i in tqdm(range(maxpoint)):
        pointdfidx = np.where(df[:,h.index('Point')] == i)
        pointdf = df[pointdfidx[0]]
        maxFrame = pointdf[:,h.index('Frame')].max() 
        idx = np.where(pointdf[:,h.index('Frame')] == maxFrame) 
        x,y = pointdf[idx[0][0],h.index('x')], pointdf[idx[0][0],h.index('y')]
        x0,y0 = pointdf[0,h.index('x')], pointdf[0,h.index('y')]
        VSL = np.sqrt(pow(x-x0,2)+pow(y-y0,2)) * microscale
        VSLlist.append([i,VSL])
    VSLframe = np.array(VSLlist, dtype='float')
    VSLframe = pd.DataFrame(VSLframe, columns=["Point", "VSL"])
    return VSLframe

#BCF算出
def makeBCF(df, h):
    dfidx = np.where(df[:,h.index('Point')] == df[:,h.index('Point')])
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    BCFlist = []
    print ('BCFを算出しています。')
    for i in tqdm(range(maxpoint)):
        BCF = 0
        pointXYidx = np.where((df[:,h.index('Point')] == i)&(df[:,h.index('area')] > 0))
        pointXY = df[pointXYidx[0]]
        pointAMidx = np.where((df[:,h.index('Point')] == i)&(df[:,h.index('Ave_x')] == df[:,h.index('Ave_x')]))
        pointAM = df[pointAMidx[0]]
        astart = 0
        for r in range(pointXY.shape[0]-1):
            ax = pointXY[r, h.index('x')]
            ay = pointXY[r, h.index('y')]
            bx = pointXY[r+1, h.index('x')]
            by = pointXY[r+1, h.index('y')]
            for a in range(pointAM.shape[0]-1):
                if a > astart:
                    cx = pointAM[a, h.index('Ave_x')]
                    cy = pointAM[a, h.index('Ave_y')]
                    dx = pointAM[a+1, h.index('Ave_x')]
                    dy = pointAM[a+1, h.index('Ave_y')]
                    ta = (cx - dx)*(ay - cy) + (cy - dy)*(cx - ax)
                    tb = (cx - dx)*(by - cy) + (cy - dy)*(cx - bx)
                    tc = (ax - bx)*(cy - ay) + (ay - by)*(ax - cx)
                    td = (ax - bx)*(dy - ay) + (ay - by)*(ax - dx)
                    if tc*td < 0:
                        if ta*tb < 0:
                            BCF = BCF + 1
                            astart = a
                            break
        BCF = BCF/2
        BCFlist.append([i,BCF])
    BCFframe = np.array(BCFlist, dtype='float')
    BCFframe = pd.DataFrame(BCFframe, columns=["Point", "BCF"])
    return BCFframe

def makeALH(df, microscale, h):
    dfidx = np.where(df[:,h.index('Point')] == df[:,h.index('Point')])
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    ALHlist = []
    print ('ALHを算出しています。')
    for p in tqdm(range(maxpoint)):
        pointdfidx = np.where((df[:,h.index('Point')] == p)&(df[:,h.index('area')] > 0))
        pointdf = df[pointdfidx[0]]
        pointAMidx = np.where((df[:,h.index('Point')] == p)&(df[:,h.index('Ave_x')] == df[:,h.index('Ave_x')]))
        pointAM = df[pointAMidx[0]]
        preLH = 100
        nowLH = 100
        postLH = 100
        totalLH = 0
        NoLH = 0
        if pointAM.shape[0] > 0:
            for i in range(pointdf.shape[0]):
                x,y = pointdf[i, h.index('x')], pointdf[i, h.index('y')]
                pointAM_LH = np.sqrt(pow(pointAM[:, h.index('Ave_x')]-x,2)+pow(pointAM[:, h.index('Ave_y')]-y,2))
                minLH = pointAM_LH.min()
                preLH = nowLH
                nowLH = postLH
                postLH = minLH
                if nowLH > preLH and nowLH > postLH:
                    totalLH = totalLH + nowLH
                    NoLH = NoLH + 1
    
        if NoLH > 0:
            ALH = totalLH / NoLH * microscale
        else:
            ALH = 0
        ALHlist.append([p,ALH])
    ALHframe = np.array(ALHlist, dtype = 'float')
    ALHframe = pd.DataFrame(ALHframe, columns=["Point", "ALH"])
    
    return ALHframe

def makeFRD(df, microscale, h):
    dfidx = np.where((df[:,h.index('Point')] == df[:,h.index('Point')])&(df[:,h.index('area')] >0))
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    FRDlist = []
    print ('Fractal Dimensionを算出しています。')
    for p in tqdm(range(maxpoint)):
        pdfidx = np.where(df[:,h.index('Point')] == p)
        pdf = df[pdfidx[0]]
        xy = pdf[:,h.index('x'):h.index('y')+1]
        xy = xy.astype(np.int64)
        (x,y),radius = cv2.minEnclosingCircle(xy)
        n = len(xy) - 1
        L = pdf[:,h.index('Length')].sum() 
        D = 0
        diam = radius*2*microscale
        if n > 0 and L > 0:
            D = math.log(n)/(math.log(n) + math.log(diam/L))
        FRDlist.append([p,diam,D])
    FRDframe = np.array(FRDlist, dtype = 'float')
    FRDframe = pd.DataFrame(FRDframe, columns=["Point",'diameter', "D"])
    return FRDframe

def makeANG(df, h):
    dfidx = np.where((df[:,h.index('Point')] == df[:,h.index('Point')]))
    df = df[dfidx[0]]
    maxpoint = int(df[:,h.index('Point')].max()) + 1
    ANGlist = []
    print ('角度を算出しています。')
    for p in tqdm(range(maxpoint)):
        pdfidx = np.where((df[:,h.index('Point')] == p)&(df[:,h.index('angle')] != 0)&(df[:,h.index('Frame')] > 6))
        pdf = df[pdfidx[0]]
        ANGs = pdf[:,h.index('angle')]
        if ANGs.shape[0] > 0:
            StdANG = np.std(ANGs)
            MeanANG = abs(ANGs.mean())
            SumANG = abs(ANGs.sum())
        else:
            SumANG = 0
            MeanANG = 0
            StdANG = 0
        
        ANGlist.append([p,SumANG,MeanANG,StdANG])

    ANGframe = np.array(ANGlist, dtype = 'float')
    ANGframe = pd.DataFrame(ANGframe, columns=["Point",'SumAngle','MeanAngle','StdAngle'])

    return ANGframe


def Add_decision_simple(df, dfs, thresh_diameter):
    
    df['motility'] = pd.Series()
    df.loc[(df['diameter'] > thresh_diameter)&(df.motility != 0),'motility'] = 1
    df.loc[~(df['diameter'] > thresh_diameter)&(df.motility != 0),'motility'] = 0

    dfs = pd.merge(dfs, df[['Point', 'motility']], on='Point', how = 'left')
    
    return df, dfs

def Add_decision_prog(df, dfs, frames, Progressive):
    
    stdheader = ['VAP']

    df2 = df.copy()
    for h in stdheader:
        df2[h] = df[h] / (df['FL_VCL'] + 1)
    
    df['prog'] = df2['VAP']*frames
    df['prog'] = df['prog'].apply(lambda x: 1 if x > Progressive else 0)

    dfs = pd.merge(dfs, df[['Point', 'prog']], on='Point', how = 'left')

    return df, dfs

def Add_decision_circle(df, dfs, frames, Circle_SumAngle, Circle_MeanAngle, Circle_StdAngle):
    df['circle'] = pd.Series()
    df.loc[(df.SumAngle >= Circle_SumAngle)&(df.MeanAngle >= Circle_MeanAngle)&(df.StdAngle <= Circle_StdAngle),'circle'] = 1
    df.loc[~((df.SumAngle >= Circle_SumAngle)&(df.MeanAngle >= Circle_MeanAngle)&(df.StdAngle <= Circle_StdAngle)),'circle'] = 0
    dfs = pd.merge(dfs, df[['Point', 'circle']], on='Point', how = 'left')

    return df, dfs

def Add_decision_derail(df,dfs,Derail_StdAngle):
    df['derail'] = pd.Series()
    df.loc[(df.StdAngle >= Derail_StdAngle),'derail'] = 1
    df.loc[~(df.StdAngle >= Derail_StdAngle),'derail'] = 0

    dfs = pd.merge(dfs, df[['Point', 'derail']], on='Point', how = 'left')

    return df, dfs
           
def SaveAllResults(tmpdf, ResultFolder):
    if os.path.exists(ResultFolder + "AllResults.csv"):
        ALLRF = pd.read_csv(ResultFolder + "AllResults.csv", header=0)
        ALLRF = pd.concat([ALLRF, tmpdf], ignore_index=True)
        ALLRF.to_csv(ResultFolder + "AllResults.csv", index = False)
    else:
        tmpdf.to_csv(ResultFolder + "AllResults.csv", index = False)

# 認識結果動画の書き出し
def writeResultMovFast(ResultFolder, filename, movarray, movBWarray, AllPoints, start_second, BW, h):
    print ('トラックを動画に書き出しています。')
    df = AllPoints

    outheight = movarray.shape[1]
    outwidth = movarray.shape[2]
    fps = movarray.shape[0]

    OL = Image.new('RGBA', (int(outwidth),int(outheight)), (0,0,0,0))
    draw = ImageDraw.Draw(OL)

    if BW == 1:
        BWflag = "_BW"
    else:
        BWflag = ""

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(ResultFolder + filename +  "_sec" + str(start_second)  + BWflag + "_track.m4v",
                             int(fourcc), fps, (int(outwidth), int(outheight)))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 軌跡ごとイメージのベース作成
    Track_base = Image.new('RGB', (int(outwidth/2),int(outheight/2)), 'white')
    draw_TL = ImageDraw.Draw(Track_base)

    for i in tqdm(range(fps)):
        if BW == 1: # BWバージョン
            img = movBWarray[i]
        else:
            img = movarray[i]
        frame = cv2.merge((img, img, img))
        
        if i > 0:
            PILframe = Image.fromarray(np.uint8(frame))
            plistdf = df[(df.Point == df.Point)&(df.Frame == i)&(df.area > 0)].reset_index(drop=True)
            fsdf = df[(df.Point == df.Point)&(df.Frame <= i)&(df.area > 0)].reset_index(drop=True)
            for p in plistdf['Point']:
                if p > 0:
                    pfsdf = fsdf[fsdf.Point == p].reset_index(drop=True)
                    mi = max(pfsdf.index)
                    if mi != 0:
                        draw.line((int(pfsdf['x'][mi-1]),int(pfsdf['y'][mi-1]),
                                    int(pfsdf['x'][mi]),int(pfsdf['y'][mi])),
                                        fill=(255,204,0), width=1)
                        # 軌跡ベース書き込み
                        draw_TL.line((int(pfsdf['x'][mi-1]),int(pfsdf['y'][mi-1]),
                                    int(pfsdf['x'][mi]),int(pfsdf['y'][mi])),
                                        fill=(200,200,2000), width=1)
                        # 
            mask = OL.split()[3]
            PILframe.paste(OL, None, mask)
            frame = np.asarray(PILframe)
        framedfdr = df[(df.Frame == i)&(df.Point == df.Point)&(df.derail == 1)&(df.area > 0)]
        for k in framedfdr.index:
            frame = cv2.putText(frame,str(int(framedfdr.loc[k,'Point'])),
                                (int(framedfdr.loc[k,'x']), int(framedfdr.loc[k,'y'])),
                                font,0.4,(255,255,255),1,cv2.LINE_AA)
        framedfim = df[(df.Frame == i)&(df.Point == df.Point)&(df.motile == 0)&(df.area > 0)]
        for k in framedfim.index:
            frame = cv2.putText(frame,str(int(framedfim.loc[k,'Point'])),
                                (int(framedfim.loc[k,'x']), int(framedfim.loc[k,'y'])),
                                font,0.4,(255,0,0),1,cv2.LINE_AA)
        framedfmobi = df[(df.Frame == i)&(df.Point == df.Point)&(df.derail == 0)&(df.motile == 1)&(df.prog == 0)&(df.area > 0)]
        for k in framedfmobi.index:
            frame = cv2.putText(frame,str(int(framedfmobi.loc[k,'Point'])),
                                (int(framedfmobi.loc[k,'x']), int(framedfmobi.loc[k,'y'])),
                                font,0.4,(0,255,255),1,cv2.LINE_AA)
        framedfprog = df[(df.Frame == i)&(df.Point == df.Point)&(df.derail == 0)&(df.motile == 1)&(df.prog == 1)&(df.area > 0)]
        for k in framedfprog.index:
            frame = cv2.putText(frame,str(int(framedfprog.loc[k,'Point'])),
                                (int(framedfprog.loc[k,'x']), int(framedfprog.loc[k,'y'])),
                                font,0.4,(0,0,255),1,cv2.LINE_AA)
        framedfcircle = df[(df.Frame == i)&(df.Point == df.Point)&(df.derail == 0)&(df.motile == 1)&(df.circle == 1)&(df.area > 0)]
        for k in framedfcircle.index:
            frame = cv2.putText(frame,str(int(framedfcircle.loc[k,'Point'])),
                                (int(framedfcircle.loc[k,'x']), int(framedfcircle.loc[k,'y'])),
                                font,0.4,(0,255,0),1,cv2.LINE_AA)

        frame = cv2.putText(frame,str(int(i)),(20,30),font,1,(0,255,0),1,cv2.LINE_AA)

        if i == fps - 1:
            cv2.imwrite(ResultFolder + filename +  "_sec" + str(start_second)  + "_lastframe.jpg", frame)
        out.write(frame)
    
    # ポイント別軌跡保存 pandas df を numpy arrayに変換する必要がある。
    font = ImageFont.truetype("arial.ttf", size=14)
    if not os.path.exists(ResultFolder + filename_with_arg + '_trackimgs'):
        os.mkdir(ResultFolder + filename_with_arg + '_trackimgs')
    Points = np.unique(df.loc[:,'Point'])
    for p in Points:
        if p > 0:
            withdata = '_frag'
            clr = 'blue'
            frames = df[df.Point == p].loc[:,'Framelength'].max()
            if frames >= max_frame:
                withdata = '_trck'
                clr = 'red'
            Track = Track_base.copy()
            dTL = ImageDraw.Draw(Track)
            #df_p_idx = np.where(df[:,h.index('Point')] == p)
            pdf = df[(df.Point == p)&(df.Framelength <= max_frame)].reset_index(drop=True)
            p_x = pdf.loc[:,'x']
            p_y = pdf.loc[:,'y']
            #dTL.text((p_x/2,p_y/2),str(pdf.loc[:,'Frame']), fill=clr, font=font)
            for f in pdf.index:
                p_pre_x = pdf.loc[f, 'pre_x']
                p_pre_y = pdf.loc[f, 'pre_y']
                p_x = pdf.loc[f, 'x']
                p_y = pdf.loc[f, 'y']
                if p_pre_x == 0 and p_pre_y == 0:
                    p_pre_x = p_x
                    p_pre_y = p_y
                dTL.line((p_pre_x/2,p_pre_y/2,p_x/2,p_y/2), fill=clr, width=3)
            #dTL.text((p_x/2,p_y/2),str(pdf.loc[:,'Frame']), fill=clr, font=font)
            Track.save(ResultFolder + filename_with_arg + '_trackimgs/' + str(p).zfill(4) + withdata + ".jpg", quality=90)
    #

    out.release()
    return 0

cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
